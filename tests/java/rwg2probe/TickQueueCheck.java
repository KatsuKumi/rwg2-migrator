package rwg2probe;

import java.io.IOException;
import java.lang.reflect.Field;
import java.util.Collection;
import java.util.concurrent.atomic.AtomicReference;

import cpw.mods.fml.common.gameevent.TickEvent;

/** Runs the probe's real TickQueue: a ticking server finishes a world op, one that stopped ticking fails it with a plain error. */
public final class TickQueueCheck {

    private static final long SILENT_NANOS = 121_000_000_000L;
    private static final long TICK_MILLIS = 20L;
    private static final long PROMPT_MILLIS = 3000L;
    private static final long LONG_STEP_MILLIS = 2500L;
    private static final int STEPS = 3;

    public static void main(String[] args) throws Exception {
        ticking();
        stalled();
        longStep();
        failing();
        System.out.println("ok");
    }

    private static TickQueue.Work<String> steps(int n) {
        return new TickQueue.Work<String>() {

            private int left = n;

            @Override
            public boolean step() {
                return --left <= 0;
            }

            @Override
            public String result() {
                return "done";
            }
        };
    }

    private static Thread server(TickQueue q) {
        Thread t = new Thread(() -> {
            while (!Thread.currentThread().isInterrupted()) {
                q.tick(new TickEvent.ServerTickEvent(TickEvent.Phase.END));
                try {
                    Thread.sleep(TICK_MILLIS);
                } catch (InterruptedException e) {
                    return;
                }
            }
        });
        t.setDaemon(true);
        t.start();
        return t;
    }

    private static void set(TickQueue q, String name, Object value) throws Exception {
        Field f = TickQueue.class.getDeclaredField(name);
        f.setAccessible(true);
        f.set(q, value);
    }

    private static int queued(TickQueue q) throws Exception {
        Field f = TickQueue.class.getDeclaredField("jobs");
        f.setAccessible(true);
        return ((Collection<?>) f.get(q)).size();
    }

    private static void check(boolean ok, String what) {
        if (!ok) throw new AssertionError(what);
    }

    private static void ticking() throws Exception {
        TickQueue q = new TickQueue();
        Thread t = server(q);
        check("done".equals(q.run(steps(STEPS))), "a ticking server finishes the op");
        t.interrupt();
        System.out.println("ticking");
    }

    private static void stalled() throws Exception {
        TickQueue q = new TickQueue();
        set(q, "beat", System.nanoTime() - SILENT_NANOS);
        long t0 = System.nanoTime();
        try {
            q.run(steps(STEPS));
            throw new AssertionError("a server silent for 121 s must fail the op");
        } catch (IllegalStateException e) {
            check(e.getMessage().contains("stopped ticking for 120 s"), "the plain error: " + e.getMessage());
        }
        check((System.nanoTime() - t0) / 1_000_000L < PROMPT_MILLIS, "it fails within a poll, not after a wait");
        check(queued(q) == 0, "the op leaves the queue");
        System.out.println("stalled");
    }

    private static void longStep() throws Exception {
        TickQueue q = new TickQueue();
        set(q, "beat", System.nanoTime() - SILENT_NANOS);
        set(q, "stepping", true);
        AtomicReference<Throwable> got = new AtomicReference<>();
        Thread caller = new Thread(() -> {
            try {
                q.run(steps(STEPS));
            } catch (Throwable e) {
                got.set(e);
            }
        });
        caller.start();
        caller.join(LONG_STEP_MILLIS);
        check(caller.isAlive() && got.get() == null, "a long step is not a stall");
        set(q, "stepping", false);
        caller.join(PROMPT_MILLIS);
        check(!caller.isAlive() && got.get() instanceof IllegalStateException, "once the step is over the silence counts");
        System.out.println("stepping");
    }

    private static void failing() throws Exception {
        TickQueue q = new TickQueue();
        Thread t = server(q);
        try {
            q.run(new TickQueue.Work<Void>() {

                @Override
                public boolean step() throws IOException {
                    throw new IOException("boom");
                }

                @Override
                public Void result() {
                    return null;
                }
            });
            throw new AssertionError("a failing op must throw");
        } catch (IOException e) {
            check("boom".equals(e.getMessage()), "the op's own error comes back");
        }
        t.interrupt();
        System.out.println("failing");
    }
}
