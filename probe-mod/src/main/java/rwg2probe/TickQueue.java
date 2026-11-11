package rwg2probe;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import cpw.mods.fml.common.eventhandler.SubscribeEvent;
import cpw.mods.fml.common.gameevent.TickEvent;

/** Runs world work on the server thread, a slice per tick, because the 1.7.10 chunk provider is single-threaded. */
public final class TickQueue {

    private static final long BUDGET_NANOS = 100_000_000L;
    private static final long STALL_NANOS = 120_000_000_000L;
    private static final long WAIT_MILLIS = 1000L;

    private final ConcurrentLinkedQueue<Job<?>> jobs = new ConcurrentLinkedQueue<>();
    private volatile long beat = System.nanoTime();
    private volatile boolean stepping;

    interface Work<T> {

        boolean step() throws Exception;

        T result();
    }

    @SubscribeEvent
    public void tick(TickEvent.ServerTickEvent event) {
        if (event.phase != TickEvent.Phase.END) return;
        beat = System.nanoTime();
        long deadline = beat + BUDGET_NANOS;
        for (Job<?> job; (job = jobs.peek()) != null && System.nanoTime() < deadline;) {
            if (step(job)) jobs.poll();
        }
    }

    private boolean step(Job<?> job) {
        stepping = true;
        try {
            return job.step();
        } finally {
            stepping = false;
            beat = System.nanoTime();
        }
    }

    /** The server thread has not ticked for STALL_NANOS outside a step: it died, hangs in its shutdown, or stopped. */
    private boolean stalled() {
        return !stepping && System.nanoTime() - beat > STALL_NANOS;
    }

    <T> T run(Work<T> work) throws Exception {
        Job<T> job = new Job<>(work);
        jobs.add(job);
        while (!stalled()) {
            try {
                return job.future.get(WAIT_MILLIS, TimeUnit.MILLISECONDS);
            } catch (TimeoutException e) {
                continue;
            } catch (ExecutionException e) {
                throw e.getCause() instanceof Exception ? (Exception) e.getCause() : e;
            }
        }
        jobs.remove(job);
        throw new IllegalStateException(
                "the server stopped ticking for " + STALL_NANOS / 1_000_000_000L + " s, so this request cannot finish");
    }

    private static final class Job<T> {

        final Work<T> work;
        final CompletableFuture<T> future = new CompletableFuture<>();

        Job(Work<T> work) {
            this.work = work;
        }

        boolean step() {
            try {
                if (!work.step()) return false;
                future.complete(work.result());
            } catch (Throwable t) {
                future.completeExceptionally(t);
            }
            return true;
        }
    }
}
