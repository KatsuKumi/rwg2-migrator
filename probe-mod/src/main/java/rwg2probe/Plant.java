package rwg2probe;

import java.util.Random;

import net.minecraft.world.WorldServer;
import net.minecraft.world.biome.BiomeGenBase;
import net.minecraft.world.gen.feature.WorldGenAbstractTree;

import com.google.gson.Gson;
import com.google.gson.JsonObject;

/**
 * plant: grows each spot's biome tree there, as the biome decorator does, in the order given, until the spot's group
 * has as many new log blocks as its budget. A spot whose chunk or one of its 8 neighbours does not exist is skipped, so
 * planting never makes the server generate a chunk.
 */
final class Plant implements TickQueue.Work<JsonObject> {

    static final int X = 0, Z = 1, BIOME = 2, GROUP = 3, WIDTH = 4;
    private static final Gson GSON = new Gson();
    private static final int REACH = 8;
    private static final int BELOW = 2;
    private static final int ABOVE = 48;
    private static final long X_FACTOR = 341873128712L;
    private static final long Z_FACTOR = 132897987541L;
    private static final long SALT = 0x5EED2000L;

    private final WorldServer world;
    private final int[][] spots;
    private final int[] budgets;
    private final int[] logs;
    private final int[] trees;
    private int index, attempts, missing, unknown;

    Plant(WorldServer world, int[][] spots, int[] budgets) {
        this.world = world;
        this.spots = spots;
        this.budgets = budgets;
        this.logs = new int[budgets.length];
        this.trees = new int[budgets.length];
    }

    @Override
    public boolean step() {
        if (index < spots.length) {
            int[] spot = spots[index++];
            if (logs[spot[GROUP]] < budgets[spot[GROUP]]) grow(spot);
        }
        return index >= spots.length;
    }

    private void grow(int[] spot) {
        int x = spot[X], z = spot[Z], group = spot[GROUP];
        BiomeGenBase[] all = BiomeGenBase.getBiomeGenArray();
        BiomeGenBase biome = spot[BIOME] >= 0 && spot[BIOME] < all.length ? all[spot[BIOME]] : null;
        if (biome == null) {
            unknown++;
            return;
        }
        if (!around(Math.floorDiv(x, WorldOps.CHUNK), Math.floorDiv(z, WorldOps.CHUNK))) {
            missing++;
            return;
        }
        int y = world.getHeightValue(x, z);
        Random rand = new Random(world.getSeed() ^ (x * X_FACTOR + z * Z_FACTOR + SALT));
        int before = wood(x, y, z);
        WorldGenAbstractTree tree = biome.func_150567_a(rand);
        tree.setScale(1.0D, 1.0D, 1.0D);
        attempts++;
        if (tree.generate(world, rand, x, y, z)) {
            tree.func_150524_b(world, rand, x, y, z);
            trees[group]++;
        }
        logs[group] += wood(x, y, z) - before;
    }

    private boolean around(int cx, int cz) {
        for (int dx = -1; dx <= 1; dx++)
            for (int dz = -1; dz <= 1; dz++) if (!WorldOps.exists(world, cx + dx, cz + dz)) return false;
        return true;
    }

    private int wood(int x, int y, int z) {
        int n = 0;
        int top = Math.min(WorldOps.HEIGHT, y + ABOVE);
        for (int i = x - REACH; i <= x + REACH; i++) for (int k = z - REACH; k <= z + REACH; k++)
            for (int j = Math.max(0, y - BELOW); j < top; j++) if (world.getBlock(i, j, k).isWood(world, i, j, k)) n++;
        return n;
    }

    @Override
    public JsonObject result() {
        JsonObject reply = new JsonObject();
        int planted = 0, placed = 0;
        for (int g = 0; g < budgets.length; g++) {
            planted += trees[g];
            placed += logs[g];
        }
        reply.addProperty("spots", spots.length);
        reply.addProperty("attempts", attempts);
        reply.addProperty("trees", planted);
        reply.addProperty("logs", placed);
        reply.addProperty("missingChunks", missing);
        reply.addProperty("unknownBiome", unknown);
        reply.add("groupTrees", GSON.toJsonTree(trees));
        reply.add("groupLogs", GSON.toJsonTree(logs));
        return reply;
    }
}
