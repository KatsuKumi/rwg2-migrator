package rwg2probe;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import net.minecraft.world.World;
import net.minecraft.world.chunk.IChunkProvider;

import com.google.common.collect.ImmutableList;

import cpw.mods.fml.common.IWorldGenerator;
import cpw.mods.fml.common.registry.GameRegistry;

/** Runs FML's world generators in a fixed order (weight, then class name), each after a World.rand seed of its own. */
final class FixedGenerators {

    private static final long SALT = 0x5EED1000L;

    private FixedGenerators() {}

    @SuppressWarnings("unchecked")
    static int install() throws ReflectiveOperationException {
        Set<IWorldGenerator> all = (Set<IWorldGenerator>) field("worldGenerators").get(null);
        Map<IWorldGenerator, Integer> weights = (Map<IWorldGenerator, Integer>) field("worldGeneratorIndex").get(null);
        List<IWorldGenerator> order = new ArrayList<>(all);
        order.sort(Comparator.<IWorldGenerator>comparingInt(weights::get).thenComparing(g -> g.getClass().getName()));
        List<IWorldGenerator> seeded = new ArrayList<>();
        for (int i = 0; i < order.size(); i++) seeded.add(new Seeded(order.get(i), SALT + i));
        field("sortedGeneratorList").set(null, ImmutableList.copyOf(seeded));
        return seeded.size();
    }

    private static Field field(String name) throws NoSuchFieldException {
        Field f = GameRegistry.class.getDeclaredField(name);
        f.setAccessible(true);
        return f;
    }

    private static final class Seeded implements IWorldGenerator {

        private final IWorldGenerator inner;
        private final long salt;

        Seeded(IWorldGenerator inner, long salt) {
            this.inner = inner;
            this.salt = salt;
        }

        @Override
        public void generate(Random random, int chunkX, int chunkZ, World world, IChunkProvider generator,
                IChunkProvider provider) {
            Reseed.seed(world, chunkX, chunkZ, salt);
            inner.generate(random, chunkX, chunkZ, world, generator, provider);
        }
    }
}
