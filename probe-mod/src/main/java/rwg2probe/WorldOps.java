package rwg2probe;

import java.nio.ByteBuffer;
import java.util.LinkedHashSet;
import java.util.Set;

import net.minecraft.block.Block;
import net.minecraft.server.MinecraftServer;
import net.minecraft.world.ChunkCoordIntPair;
import net.minecraft.world.WorldServer;
import net.minecraft.world.chunk.Chunk;
import net.minecraft.world.chunk.storage.AnvilChunkLoader;
import net.minecraft.world.gen.ChunkProviderServer;
import net.minecraft.world.storage.ThreadedFileIOBase;
import net.minecraftforge.common.DimensionManager;

import com.google.gson.JsonObject;

/** generate, save, plant and chunk: the chunk work of a gen boot, always on the server thread through the TickQueue. */
final class WorldOps {

    static final int CHUNK = 16;
    static final int HEIGHT = 256;
    static final int VOXELS = CHUNK * CHUNK * HEIGHT;
    static final int COLUMNS = CHUNK * CHUNK;
    static final int DUMP_BYTES = VOXELS * (Integer.BYTES + Short.BYTES) + COLUMNS * Short.BYTES;

    private final TickQueue queue;

    WorldOps(TickQueue queue) {
        this.queue = queue;
    }

    JsonObject generate(long[] chunks) throws Exception {
        return queue.run(new Generate(overworld(), chunks));
    }

    JsonObject save() throws Exception {
        JsonObject reply = queue.run(new Save());
        ThreadedFileIOBase.threadedIOInstance.waitForFinish();
        reply.addProperty("flushed", true);
        return reply;
    }

    JsonObject plant(int[][] spots, int[] budgets) throws Exception {
        return queue.run(new Plant(overworld(), spots, budgets));
    }

    byte[] chunk(int x, int z, JsonObject reply) throws Exception {
        return queue.run(new Dump(overworld(), x, z, reply));
    }

    static long pack(int x, int z) {
        return ChunkCoordIntPair.chunkXZ2Int(x, z);
    }

    private static int x(long key) {
        return (int) key;
    }

    private static int z(long key) {
        return (int) (key >> Integer.SIZE);
    }

    private static WorldServer overworld() {
        WorldServer world = DimensionManager.getWorld(0);
        if (world == null) throw new IllegalStateException("the overworld is not loaded");
        return world;
    }

    static boolean exists(WorldServer world, int x, int z) {
        ChunkProviderServer provider = world.theChunkProviderServer;
        if (provider.chunkExists(x, z)) return true;
        return provider.currentChunkLoader instanceof AnvilChunkLoader
                && ((AnvilChunkLoader) provider.currentChunkLoader).chunkExists(world, x, z);
    }

    /** Loads or generates each chunk, then populates every 2x2 group whose four chunks now exist. */
    private static final class Generate implements TickQueue.Work<JsonObject> {

        private final WorldServer world;
        private final ChunkProviderServer provider;
        private final long[] chunks;
        private final Set<Long> origins = new LinkedHashSet<>();
        private Long[] pending;
        private int index, existed, generated, populated;

        Generate(WorldServer world, long[] chunks) {
            this.world = world;
            this.provider = world.theChunkProviderServer;
            this.chunks = chunks;
        }

        @Override
        public boolean step() {
            if (index < chunks.length) {
                load(chunks[index++]);
                return false;
            }
            if (pending == null) pending = origins.toArray(new Long[0]);
            int i = index++ - chunks.length;
            if (i < pending.length) populate(x(pending[i]), z(pending[i]));
            return i + 1 >= pending.length;
        }

        private void load(long key) {
            int x = x(key), z = z(key);
            if (exists(world, x, z)) existed++;
            else generated++;
            provider.loadChunk(x, z);
            for (int dx = -1; dx <= 0; dx++) for (int dz = -1; dz <= 0; dz++) origins.add(pack(x + dx, z + dz));
        }

        private void populate(int x, int z) {
            for (int dx = 0; dx <= 1; dx++) for (int dz = 0; dz <= 1; dz++) if (!exists(world, x + dx, z + dz)) return;
            for (int dx = 0; dx <= 1; dx++) for (int dz = 0; dz <= 1; dz++) provider.loadChunk(x + dx, z + dz);
            if (provider.provideChunk(x, z).isTerrainPopulated) return;
            provider.populate(provider, x, z);
            populated++;
        }

        @Override
        public JsonObject result() {
            JsonObject reply = new JsonObject();
            reply.addProperty("requested", chunks.length);
            reply.addProperty("existed", existed);
            reply.addProperty("generated", generated);
            reply.addProperty("populatedByProbe", populated);
            reply.addProperty("loadedChunks", provider.getLoadedChunkCount());
            return reply;
        }
    }

    private static final class Save implements TickQueue.Work<JsonObject> {

        private int worlds;

        @Override
        public boolean step() throws Exception {
            for (WorldServer world : MinecraftServer.getServer().worldServers) {
                world.saveAllChunks(true, null);
                worlds++;
            }
            return true;
        }

        @Override
        public JsonObject result() {
            JsonObject reply = new JsonObject();
            reply.addProperty("worlds", worlds);
            return reply;
        }
    }

    private static final class Dump implements TickQueue.Work<byte[]> {

        private final WorldServer world;
        private final int x, z;
        private final JsonObject reply;
        private byte[] data;

        Dump(WorldServer world, int x, int z, JsonObject reply) {
            this.world = world;
            this.x = x;
            this.z = z;
            this.reply = reply;
        }

        @Override
        public boolean step() {
            if (!exists(world, x, z)) throw new IllegalArgumentException("chunk " + x + "," + z + " does not exist");
            Chunk chunk = world.theChunkProviderServer.loadChunk(x, z);
            ByteBuffer out = ByteBuffer.allocate(DUMP_BYTES);
            for (int y = 0; y < HEIGHT; y++) for (int lz = 0; lz < CHUNK; lz++)
                for (int lx = 0; lx < CHUNK; lx++) out.putInt(Block.getIdFromBlock(chunk.getBlock(lx, y, lz)));
            for (int y = 0; y < HEIGHT; y++) for (int lz = 0; lz < CHUNK; lz++)
                for (int lx = 0; lx < CHUNK; lx++) out.putShort((short) chunk.getBlockMetadata(lx, y, lz));
            for (int lz = 0; lz < CHUNK; lz++) for (int lx = 0; lx < CHUNK; lx++)
                out.putShort((short) chunk.getBiomeGenForWorldCoords(lx, lz, world.getWorldChunkManager()).biomeID);
            reply.addProperty("populated", chunk.isTerrainPopulated);
            data = out.array();
            return true;
        }

        @Override
        public byte[] result() {
            return data;
        }
    }
}
