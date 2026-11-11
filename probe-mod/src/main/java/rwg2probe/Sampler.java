package rwg2probe;

import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;

import rwg2.biomes.realistic.RealisticBiomeBase;
import rwg2.biomes.realistic.ocean.RealisticBiomeOcean;
import rwg2.config.ConfigRWG;
import rwg2.world.ChunkGeneratorRealistic;
import rwg2.world.ChunkManagerRealistic;

/** Heights and biomes from fresh RWG2 generator objects, one set per worker thread; no world is touched. */
final class Sampler implements AutoCloseable {

    static final int RECORD_BYTES = 8;
    private static final int CHUNK = 16;
    private static final int COLUMNS = CHUNK * CHUNK;
    private static final MethodHandle META_BIOME = metaBiome();

    private final long seed;
    private final boolean continental;
    private final int threads;
    private final ExecutorService pool;
    private final Map<String, Short> palette = new ConcurrentHashMap<>();
    private final AtomicInteger nextPalette = new AtomicInteger();

    Sampler(long seed, boolean continental, int threads) {
        if (META_BIOME == null) throw new IllegalStateException(
                "this rwg2 version has no ChunkManagerRealistic.getMetaBiomeAt(int, int); the probe needs an update");
        this.seed = seed;
        this.continental = continental;
        this.threads = threads;
        this.pool = Executors.newFixedThreadPool(threads);
    }

    private static MethodHandle metaBiome() {
        try {
            Method method = ChunkManagerRealistic.class.getDeclaredMethod("getMetaBiomeAt", int.class, int.class);
            method.setAccessible(true);
            return MethodHandles.lookup().unreflect(method);
        } catch (ReflectiveOperationException | RuntimeException e) {
            return null;
        }
    }

    @Override
    public void close() {
        pool.shutdownNow();
    }

    List<String> palette() {
        String[] names = new String[palette.size()];
        for (Map.Entry<String, Short> entry : palette.entrySet()) names[entry.getValue()] = entry.getKey();
        return Arrays.asList(names);
    }

    byte[] sample(int[][] configs, int[][] points) throws Exception {
        PointChunks chunks = new PointChunks(points);
        ByteBuffer out = ByteBuffer.allocate(configs.length * points.length * RECORD_BYTES);
        int[] saved = offsets();
        try {
            for (int[] config : configs) {
                setOffsets(config);
                Records records = new Records(points.length);
                parallel(chunks.count(), (worker, chunk) -> samplePointChunk(worker, chunks, chunk, records));
                records.writeTo(out);
            }
        } finally {
            setOffsets(saved);
        }
        return out.array();
    }

    byte[] render(int[] config, Window window) throws Exception {
        int res = window.size / window.bpp;
        ByteBuffer out = ByteBuffer.allocate(res * res * RECORD_BYTES);
        int[] saved = offsets();
        try {
            setOffsets(config);
            Records records = new Records(res * res);
            int side = window.size / CHUNK;
            parallel(side * side, (worker, chunk) -> renderChunk(worker, window, chunk, records));
            records.writeTo(out);
        } finally {
            setOffsets(saved);
        }
        return out.array();
    }

    private void samplePointChunk(Worker worker, PointChunks chunks, int chunk, Records records) {
        int[] first = chunks.point(chunks.start(chunk));
        float[] heights = worker
                .noise(Math.floorDiv(first[0], CHUNK) * CHUNK, Math.floorDiv(first[1], CHUNK) * CHUNK, 1);
        for (int i = chunks.start(chunk); i < chunks.start(chunk + 1); i++) {
            int[] p = chunks.point(i);
            records.put(chunks.index(i), worker.column(heights, p[0], p[1]));
        }
    }

    private void renderChunk(Worker worker, Window window, int chunk, Records records) {
        int side = window.size / CHUNK;
        int baseX = window.x0 + (chunk % side) * CHUNK, baseZ = window.z0 + (chunk / side) * CHUNK;
        int res = window.size / window.bpp;
        float[] heights = worker.noise(baseX, baseZ, window.bpp);
        for (int z = 0; z < CHUNK; z += window.bpp) {
            for (int x = 0; x < CHUNK; x += window.bpp) {
                int index = ((baseZ - window.z0 + z) / window.bpp) * res + (baseX - window.x0 + x) / window.bpp;
                records.put(index, worker.column(heights, baseX + x, baseZ + z));
            }
        }
    }

    private void parallel(int tasks, ChunkTask task) throws Exception {
        AtomicInteger next = new AtomicInteger();
        List<Future<?>> jobs = new ArrayList<>();
        for (int t = 0; t < threads; t++) jobs.add(pool.submit(() -> {
            Worker worker = new Worker();
            for (int c; (c = next.getAndIncrement()) < tasks;) task.run(worker, c);
            return null;
        }));
        for (Future<?> job : jobs) job.get();
    }

    private short paletteIndex(String name) {
        return palette.computeIfAbsent(name, k -> (short) nextPalette.getAndIncrement());
    }

    static int[] offsets() {
        return new int[] { ConfigRWG.biomeOffsetX, ConfigRWG.biomeOffsetZ, ConfigRWG.landmassOffsetX,
                ConfigRWG.landmassOffsetZ };
    }

    private static void setOffsets(int[] config) {
        ConfigRWG.biomeOffsetX = config[0];
        ConfigRWG.biomeOffsetZ = config[1];
        ConfigRWG.landmassOffsetX = config[2];
        ConfigRWG.landmassOffsetZ = config[3];
    }

    private interface ChunkTask {

        void run(Worker worker, int chunk);
    }

    static final class Window {

        final int x0, z0, size, bpp;

        Window(int x0, int z0, int size, int bpp) {
            this.x0 = x0;
            this.z0 = z0;
            this.size = size;
            this.bpp = bpp;
        }
    }

    private final class Worker {

        private final ChunkManagerRealistic manager = new ChunkManagerRealistic(seed, continental);
        private final ChunkGeneratorRealistic generator = new ChunkGeneratorRealistic(manager, seed, continental);
        private final RealisticBiomeBase[] biomes = new RealisticBiomeBase[COLUMNS];

        float[] noise(int x, int z, int stride) {
            return generator.getNewNoise(manager, x, z, biomes, stride);
        }

        private int metaBiome(int x, int z) {
            try {
                return (int) META_BIOME.invokeExact(manager, x, z);
            } catch (Throwable t) {
                throw new IllegalStateException("ChunkManagerRealistic.getMetaBiomeAt failed", t);
            }
        }

        Column column(float[] heights, int x, int z) {
            int localX = Math.floorMod(x, CHUNK), localZ = Math.floorMod(z, CHUNK);
            RealisticBiomeBase biome = biomes[localZ * CHUNK + localX];
            return new Column(
                    heights[localX * CHUNK + localZ],
                    paletteIndex(biome.baseBiome.biomeName),
                    (byte) metaBiome(x, z),
                    biome instanceof RealisticBiomeOcean);
        }
    }

    private static final class Column {

        final float height;
        final short biome;
        final byte meta;
        final boolean ocean;

        Column(float height, short biome, byte meta, boolean ocean) {
            this.height = height;
            this.biome = biome;
            this.meta = meta;
            this.ocean = ocean;
        }
    }

    private static final class Records {

        private final float[] height;
        private final short[] biome;
        private final byte[] meta;
        private final byte[] ocean;

        Records(int n) {
            height = new float[n];
            biome = new short[n];
            meta = new byte[n];
            ocean = new byte[n];
        }

        void put(int i, Column c) {
            height[i] = c.height;
            biome[i] = c.biome;
            meta[i] = c.meta;
            ocean[i] = (byte) (c.ocean ? 1 : 0);
        }

        void writeTo(ByteBuffer out) {
            for (int i = 0; i < height.length; i++) {
                out.putFloat(height[i]).putShort(biome[i]).put(meta[i]).put(ocean[i]);
            }
        }
    }

    private static final class PointChunks {

        private final int[][] points;
        private final Integer[] order;
        private final int[] starts;

        PointChunks(int[][] points) {
            this.points = points;
            order = new Integer[points.length];
            for (int i = 0; i < points.length; i++) order[i] = i;
            Arrays.sort(order, (a, b) -> Long.compare(key(points[a]), key(points[b])));
            List<Integer> s = new ArrayList<>();
            for (int i = 0; i < order.length; i++) {
                if (i == 0 || key(points[order[i]]) != key(points[order[i - 1]])) s.add(i);
            }
            s.add(order.length);
            starts = new int[s.size()];
            for (int i = 0; i < starts.length; i++) starts[i] = s.get(i);
        }

        int count() {
            return starts.length - 1;
        }

        int start(int chunk) {
            return starts[chunk];
        }

        int index(int sorted) {
            return order[sorted];
        }

        int[] point(int sorted) {
            return points[order[sorted]];
        }

        private static long key(int[] p) {
            return ((long) Math.floorDiv(p[0], CHUNK) << 32) | (Math.floorDiv(p[1], CHUNK) & 0xffffffffL);
        }
    }
}
