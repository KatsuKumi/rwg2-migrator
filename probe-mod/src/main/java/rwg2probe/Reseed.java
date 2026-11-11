package rwg2probe;

import net.minecraft.world.World;
import net.minecraftforge.event.terraingen.PopulateChunkEvent;

import cpw.mods.fml.common.eventhandler.SubscribeEvent;

/** Seeds World.rand from the world seed and the chunk at the start and end of RWG2's own decoration. */
public final class Reseed {

    private static final long X_FACTOR = 341873128712L;
    private static final long Z_FACTOR = 132897987541L;
    private static final long PRE_SALT = 0x5EED0001L;
    private static final long POST_SALT = 0x5EED0002L;

    @SubscribeEvent
    public void pre(PopulateChunkEvent.Pre event) {
        seed(event.world, event.chunkX, event.chunkZ, PRE_SALT);
    }

    @SubscribeEvent
    public void post(PopulateChunkEvent.Post event) {
        seed(event.world, event.chunkX, event.chunkZ, POST_SALT);
    }

    static void seed(World world, int x, int z, long salt) {
        world.rand.setSeed(world.getSeed() ^ (x * X_FACTOR + z * Z_FACTOR + salt));
    }
}
