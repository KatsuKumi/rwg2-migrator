package rwg2probe;

import java.util.Optional;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import net.minecraft.server.MinecraftServer;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import cpw.mods.fml.common.FMLCommonHandler;

/** Stops the server when the tool that started it exits, or when nobody has asked anything for a while. */
final class Lifecycle {

    private static final Logger LOG = LogManager.getLogger("rwg2probe");
    private static final long POLL_MILLIS = 1000L;
    private static final double NANOS_PER_MINUTE = 60e9;
    private static final int EXIT_BEFORE_START = 3;
    private static final int EXIT_HUNG = 4;
    private static final long HALT_GRACE_MILLIS = 50_000L;
    private static final AtomicBoolean HALT_ARMED = new AtomicBoolean();
    private static final AtomicLong LAST_ACTIVITY = new AtomicLong(System.nanoTime());
    private static final AtomicInteger BUSY = new AtomicInteger();
    private static volatile boolean started;

    private Lifecycle() {}

    static void watchParent(long pid) {
        if (pid <= 0) return;
        try {
            Class<?> type = Class.forName("java.lang.ProcessHandle");
            Optional<?> handle = (Optional<?>) type.getMethod("of", long.class).invoke(null, pid);
            if (!handle.isPresent()) {
                shutdown("the tool (pid " + pid + ") is not running");
                return;
            }
            CompletableFuture<?> exit = (CompletableFuture<?>) type.getMethod("onExit").invoke(handle.get());
            exit.thenRun(() -> shutdown("the tool (pid " + pid + ") exited"));
            LOG.info("rwg2probe stops when the tool (pid {}) exits", pid);
        } catch (ReflectiveOperationException e) {
            LOG.warn("rwg2probe cannot watch the tool's process; only the idle timeout applies", e);
        }
    }

    static void watchIdle(double minutes) {
        started = true;
        touch();
        if (minutes <= 0) return;
        long limit = (long) (minutes * NANOS_PER_MINUTE);
        Thread thread = new Thread(() -> idleLoop(limit, minutes), "rwg2probe-idle");
        thread.setDaemon(true);
        thread.start();
    }

    private static void idleLoop(long limit, double minutes) {
        while (BUSY.get() > 0 || System.nanoTime() - LAST_ACTIVITY.get() < limit) {
            try {
                Thread.sleep(POLL_MILLIS);
            } catch (InterruptedException e) {
                return;
            }
        }
        shutdown("no request for " + minutes + " minutes");
    }

    static void begin() {
        BUSY.incrementAndGet();
        touch();
    }

    static void end() {
        touch();
        BUSY.decrementAndGet();
    }

    private static void touch() {
        LAST_ACTIVITY.set(System.nanoTime());
    }

    static void shutdown(String reason) {
        LOG.info("rwg2probe stops the server: {}", reason);
        MinecraftServer server = MinecraftServer.getServer();
        if (!started || server == null) {
            FMLCommonHandler.instance().exitJava(EXIT_BEFORE_START, true);
            return;
        }
        server.initiateShutdown();
        if (HALT_ARMED.compareAndSet(false, true)) haltIfStillAlive();
    }

    /**
     * A dead server thread or a shutdown stuck on a dead save thread never ends the JVM; this daemon halts it instead,
     * through FML, whose TerminalTransformer traps a mod's own Runtime.halt.
     */
    private static void haltIfStillAlive() {
        Thread thread = new Thread(() -> {
            try {
                Thread.sleep(HALT_GRACE_MILLIS);
            } catch (InterruptedException e) {
                return;
            }
            LOG.warn("rwg2probe halts the JVM: the server did not stop within {} s", HALT_GRACE_MILLIS / 1000);
            FMLCommonHandler.instance().exitJava(EXIT_HUNG, true);
        }, "rwg2probe-halt");
        thread.setDaemon(true);
        thread.start();
    }
}
