# rwg2probe

A server-side Forge 1.7.10 mod that runs inside an unmodified official GTNH dedicated server and serves the RWG2
generator on a localhost TCP port. rwg2-migrator unpacks the official server pack, drops `rwg2probe-<version>.jar` into
`mods/` and boots it headless; users never compile Minecraft code.

## Build

```bash
VERSION=0.3.1 JAVA_HOME=<JDK 17+> ./gradlew --offline spotlessApply build   # GTNH buildscript; compileOnly RWG2 2.0.6:dev
cp build/libs/rwg2probe-0.3.1.jar ../rwg2migrator/data/
```

Only the public RWG2 2.0.6 API is compiled against. `ChunkManagerRealistic.getMetaBiomeAt` is package-private, so it is
called through a method handle; a changed signature makes `sample` and `render` fail with a clear error, which
`generator check` reports.

## Launch settings

| JVM property | Meaning |
|---|---|
| `rwg2probe.port` | TCP port on 127.0.0.1; `0` picks a free one. The port is written to `rwg2probe.port` in the server folder. |
| `rwg2probe.instance` | marker the tool uses to recognise its own process; returned by `status` |
| `rwg2probe.kind` | `sampler` (FLAT throwaway world: `sample`, `render`) or `gen` (a world copy: `generate`, `save`, `plant`, `chunk`) |
| `rwg2probe.parentPid` | the tool's pid; the server stops when that process exits (`ProcessHandle.onExit`) |
| `rwg2probe.idleMinutes` | the server stops after this long without a request (default 5; 0 turns it off) |

The session token is read from `rwg2probe.token` in the server folder, never from the command line. Before the server
has started, a stop request from the parent watch halts the JVM at once; after it, the server saves and exits normally.

## Protocol

One JSON object per line in, one per line out, one request at a time. Every request carries `token`; a missing or wrong
token gets `ok:false` and the connection stays open. Every reply has `ok`, `op` and `seconds`, or `error` when `ok` is
false.

| op | boot | request | reply |
|---|---|---|---|
| `status` | both | | `instance`, `kind`, `rwg2`, `probe`, `noise`, `mods`, `biomes`, `java`, `maxHeapMB`, `usedHeapMB`, `threads`, `configOffsets`, `worldType` |
| `biomes` | both | | `biomes`: id, name, temperature, rainfall, top and filler block of every registered biome |
| `stop` | both | | `stopping`; then the server saves and exits |
| `sample` | sampler | `seed` (string), `continental`, `configs` `[[bx,bz,lx,lz],...]`, `points` `[[x,z],...]`, optional `threads` | `n`, `m`, `palette`, `data` |
| `render` | sampler | `seed`, `continental`, `config`, `x0`, `z0`, `size` (chunk-aligned), `bpp` (1/2/4/8/16), optional `threads` | `res`, `palette`, `data` |
| `generate` | gen | `offsets` (must equal rwg2.cfg), `chunks` `[[x,z],...]` | `requested`, `existed`, `generated`, `populatedByProbe`, `loadedChunks` |
| `save` | gen | | `worlds`, `flushed` (after `ThreadedFileIOBase` has drained) |
| `plant` | gen | `spots` `[[x,z,biome,group],...]`, `budgets` `[logs,...]` (one per group) | `spots`, `attempts`, `trees`, `logs`, `missingChunks`, `unknownBiome`, `groupTrees`, `groupLogs` |
| `chunk` | gen | `x`, `z` (an existing chunk) | `populated`, `data` |

- `sample`/`render` `data`: base64 of big-endian 8-byte records: f32 height (`getNewNoise`), i16 index into `palette`
  (RWG2 biome `baseBiome.biomeName`), u8 meta biome, u8 ocean flag; config-major for `sample`, row-major for `render`.
  A reply is capped at 256 MB.
- `chunk` `data`: big-endian i32 block ids then u16 meta, both in y, z, x order (65,536 each), then 256 u16 biome ids
  in z, x order.
- `plant` grows, at each spot in order, the tree its biome's decorator would (`BiomeGenBase.func_150567_a`, placed at
  `getHeightValue`, with a `Random` seeded from the world seed and the spot), until the spot's group holds `budget` new
  log blocks (counted with `Block.isWood` within 8 blocks of the spot). A spot whose chunk or one of its 8 neighbours
  does not exist is skipped, so planting never generates a chunk.
- `generate`, `save`, `plant` and `chunk` run on the server thread through a tick queue (100 ms of work per tick). `generate`
  loads or generates each chunk, then populates every 2x2 group whose four chunks exist, so population does not depend
  on which chunks the server's autosave has unloaded in between.
- In a gen boot, decoration is made repeatable. FML keeps its world generators in a `HashSet`, so generators of equal
  weight run in a different order in each JVM; the probe replaces FML's sorted list with one ordered by weight, then
  class name. `World.rand`, which server ticks also draw from, gets a fixed seed from the world seed and the chunk
  position before each of those generators, and at `PopulateChunkEvent.Pre`/`Post` around RWG2's own decoration
  (Et Futurum's deepslate edge draws `nextInt(4)` per column from `World.rand`).
- Sampling refuses a world whose RWG noise is not Perlin (`UNKNOWN`/`DYNAMICPERLIN`).
