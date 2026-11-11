package rwg2probe;

import java.lang.management.ManagementFactory;
import java.lang.management.MemoryUsage;
import java.util.Arrays;

import net.minecraft.block.Block;
import net.minecraft.world.WorldServer;
import net.minecraft.world.biome.BiomeGenBase;
import net.minecraftforge.common.DimensionManager;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;

import cpw.mods.fml.common.Loader;
import rwg2.util.NoiseImplementation;
import rwg2.world.RwgWorldSavedData;

/** Checks the token and the boot kind, then runs one request. */
final class Ops {

    static final String SAMPLER = "sampler";
    static final String GEN = "gen";
    private static final Gson GSON = new Gson();
    private static final long MAX_REPLY_BYTES = 1L << 28;
    private static final int[] STRIDES = { 1, 2, 4, 8, 16 };
    private static final int OFFSETS = 4;
    private static final int MEGABYTE_SHIFT = 20;

    private final Token token;
    private final String kind = System.getProperty("rwg2probe.kind", SAMPLER);
    private final WorldOps world;

    Ops(Token token, TickQueue queue) {
        this.token = token;
        this.world = new WorldOps(queue);
    }

    byte[] run(String op, JsonObject request, JsonObject reply) throws Exception {
        token.check(request.has("token") ? request.get("token").getAsString() : null);
        switch (op) {
            case "status":
                status(reply);
                return null;
            case "biomes":
                reply.add("biomes", biomes());
                return null;
            case "stop":
                reply.addProperty("stopping", true);
                return null;
            case "sample":
                return sample(require(SAMPLER, op, request), reply);
            case "render":
                return render(require(SAMPLER, op, request), reply);
            case "generate":
                return merge(reply, world.generate(chunks(require(GEN, op, request))));
            case "save":
                require(GEN, op, request);
                return merge(reply, world.save());
            case "plant":
                require(GEN, op, request);
                return merge(reply, plant(request));
            case "chunk":
                require(GEN, op, request);
                return world.chunk(request.get("x").getAsInt(), request.get("z").getAsInt(), reply);
            default:
                throw new IllegalArgumentException("unknown op " + op);
        }
    }

    private JsonObject require(String wanted, String op, JsonObject request) {
        if (!kind.equals(wanted))
            throw new IllegalStateException(op + " needs a " + wanted + " boot; this is a " + kind + " boot");
        return request;
    }

    private static byte[] merge(JsonObject reply, JsonObject part) {
        part.entrySet().forEach(e -> reply.add(e.getKey(), e.getValue()));
        return null;
    }

    private void status(JsonObject reply) {
        reply.addProperty("instance", System.getProperty("rwg2probe.instance", ""));
        reply.addProperty("kind", kind);
        reply.addProperty("rwg2", Loader.instance().getIndexedModList().get("rwg2").getVersion());
        reply.addProperty("probe", Tags.VERSION);
        reply.addProperty("noise", RwgWorldSavedData.getNoiseImplementation().name());
        reply.addProperty("mods", Loader.instance().getActiveModList().size());
        reply.addProperty(
                "biomes",
                (int) Arrays.stream(BiomeGenBase.getBiomeGenArray()).filter(b -> b != null).count());
        reply.addProperty("java", System.getProperty("java.version"));
        MemoryUsage heap = ManagementFactory.getMemoryMXBean().getHeapMemoryUsage();
        reply.addProperty("maxHeapMB", heap.getMax() >> MEGABYTE_SHIFT);
        reply.addProperty("usedHeapMB", heap.getUsed() >> MEGABYTE_SHIFT);
        reply.addProperty("threads", Runtime.getRuntime().availableProcessors());
        reply.add("configOffsets", GSON.toJsonTree(Sampler.offsets()));
        WorldServer overworld = DimensionManager.getWorld(0);
        if (overworld != null)
            reply.addProperty("worldType", overworld.getWorldInfo().getTerrainType().getWorldTypeName());
    }

    private static JsonArray biomes() {
        JsonArray out = new JsonArray();
        for (BiomeGenBase biome : BiomeGenBase.getBiomeGenArray()) {
            if (biome == null) continue;
            JsonObject b = new JsonObject();
            b.addProperty("id", biome.biomeID);
            b.addProperty("name", biome.biomeName);
            b.addProperty("temperature", biome.temperature);
            b.addProperty("rainfall", biome.rainfall);
            b.addProperty("top", blockName(biome.topBlock));
            b.addProperty("filler", blockName(biome.fillerBlock));
            out.add(b);
        }
        return out;
    }

    private static String blockName(Block block) {
        return block == null ? null : Block.blockRegistry.getNameForObject(block);
    }

    private static long[] chunks(JsonObject request) {
        int[] offsets = ints(request.getAsJsonArray("offsets"), OFFSETS, "offsets");
        if (!Arrays.equals(offsets, Sampler.offsets())) throw new IllegalStateException(
                "offsets " + Arrays.toString(offsets) + " differ from rwg2.cfg " + Arrays.toString(Sampler.offsets()));
        int[][] cells = ints(request.getAsJsonArray("chunks"), 2);
        long[] out = new long[cells.length];
        for (int i = 0; i < cells.length; i++) out[i] = WorldOps.pack(cells[i][0], cells[i][1]);
        return out;
    }

    private JsonObject plant(JsonObject request) throws Exception {
        int[][] spots = ints(request.getAsJsonArray("spots"), Plant.WIDTH);
        JsonArray groups = request.getAsJsonArray("budgets");
        int[] budgets = ints(groups, groups.size(), "budgets");
        for (int[] spot : spots) if (spot[Plant.GROUP] < 0 || spot[Plant.GROUP] >= budgets.length)
            throw new IllegalArgumentException("spot group " + spot[Plant.GROUP] + " has no budget");
        return world.plant(spots, budgets);
    }

    private static byte[] sample(JsonObject request, JsonObject reply) throws Exception {
        int[][] configs = ints(request.getAsJsonArray("configs"), OFFSETS);
        int[][] points = ints(request.getAsJsonArray("points"), 2);
        if (configs.length == 0 || points.length == 0)
            throw new IllegalArgumentException("configs and points are required");
        if ((long) configs.length * points.length * Sampler.RECORD_BYTES > MAX_REPLY_BYTES)
            throw new IllegalArgumentException("reply too large; split the configs into smaller requests");
        try (Sampler sampler = sampler(request, reply)) {
            byte[] data = sampler.sample(configs, points);
            reply.addProperty("n", points.length);
            reply.addProperty("m", configs.length);
            reply.add("palette", GSON.toJsonTree(sampler.palette()));
            return data;
        }
    }

    private static byte[] render(JsonObject request, JsonObject reply) throws Exception {
        int[] config = ints(request.getAsJsonArray("config"), OFFSETS, "config");
        Sampler.Window window = window(request);
        try (Sampler sampler = sampler(request, reply)) {
            byte[] data = sampler.render(config, window);
            reply.addProperty("res", window.size / window.bpp);
            reply.add("palette", GSON.toJsonTree(sampler.palette()));
            return data;
        }
    }

    private static Sampler.Window window(JsonObject request) {
        int x0 = request.get("x0").getAsInt(), z0 = request.get("z0").getAsInt();
        int size = request.get("size").getAsInt(), bpp = request.get("bpp").getAsInt();
        if (Math.floorMod(x0, WorldOps.CHUNK) != 0 || Math.floorMod(z0, WorldOps.CHUNK) != 0
                || size <= 0
                || size % WorldOps.CHUNK != 0)
            throw new IllegalArgumentException("x0, z0 and size must be positive multiples of 16 (chunk-aligned)");
        if (Arrays.stream(STRIDES).noneMatch(s -> s == bpp))
            throw new IllegalArgumentException("bpp must be 1, 2, 4, 8 or 16");
        if ((long) size / bpp * (size / bpp) * Sampler.RECORD_BYTES > MAX_REPLY_BYTES)
            throw new IllegalArgumentException("render too large; use a coarser bpp or tiles");
        return new Sampler.Window(x0, z0, size, bpp);
    }

    private static Sampler sampler(JsonObject request, JsonObject reply) {
        long seed = Long.parseLong(request.get("seed").getAsString());
        boolean continental = !request.has("continental") || request.get("continental").getAsBoolean();
        int threads = request.has("threads") ? request.get("threads").getAsInt()
                : Runtime.getRuntime().availableProcessors();
        NoiseImplementation noise = RwgWorldSavedData.getNoiseImplementation();
        if (noise != NoiseImplementation.UNKNOWN && noise != NoiseImplementation.DYNAMICPERLIN)
            throw new IllegalStateException("probe world uses " + noise + " noise; migrated worlds use Perlin");
        reply.addProperty("seed", Long.toString(seed));
        reply.addProperty("continental", continental);
        reply.addProperty("noise", noise.name());
        reply.addProperty("threads", threads);
        return new Sampler(seed, continental, threads);
    }

    private static int[][] ints(JsonArray rows, int width) {
        int[][] out = new int[rows.size()][];
        for (int i = 0; i < out.length; i++) out[i] = ints(rows.get(i).getAsJsonArray(), width, "row " + i);
        return out;
    }

    private static int[] ints(JsonArray row, int width, String what) {
        if (row.size() != width) throw new IllegalArgumentException(what + " needs " + width + " integers");
        int[] out = new int[width];
        for (int i = 0; i < width; i++) out[i] = row.get(i).getAsInt();
        return out;
    }
}
