package cpw.mods.fml.common.gameevent;

/** Stand-in for FML's TickEvent: only the phase TickQueue reads. */
public class TickEvent {

    public enum Phase {
        START,
        END
    }

    public final Phase phase;

    public TickEvent(Phase phase) {
        this.phase = phase;
    }

    public static class ServerTickEvent extends TickEvent {

        public ServerTickEvent(Phase phase) {
            super(phase);
        }
    }
}
