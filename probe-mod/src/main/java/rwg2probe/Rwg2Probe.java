package rwg2probe;

import java.io.File;
import java.io.IOException;

import net.minecraftforge.common.MinecraftForge;

import org.apache.logging.log4j.LogManager;

import cpw.mods.fml.common.FMLCommonHandler;
import cpw.mods.fml.common.Mod;
import cpw.mods.fml.common.Mod.EventHandler;
import cpw.mods.fml.common.event.FMLConstructionEvent;
import cpw.mods.fml.common.event.FMLServerAboutToStartEvent;
import cpw.mods.fml.common.event.FMLServerStartedEvent;

@Mod(
        modid = "rwg2probe",
        name = "RWG2 Probe",
        version = Tags.VERSION,
        dependencies = "required-after:rwg2",
        acceptableRemoteVersions = "*")
public class Rwg2Probe {

    static final String PORT_FILE = "rwg2probe.port";
    static final String TOKEN_FILE = "rwg2probe.token";

    @EventHandler
    public void construct(FMLConstructionEvent event) {
        Lifecycle.watchParent(Long.getLong("rwg2probe.parentPid", 0L));
        if (Ops.GEN.equals(System.getProperty("rwg2probe.kind"))) MinecraftForge.EVENT_BUS.register(new Reseed());
    }

    @EventHandler
    public void aboutToStart(FMLServerAboutToStartEvent event) throws ReflectiveOperationException {
        if (Ops.GEN.equals(System.getProperty("rwg2probe.kind"))) LogManager.getLogger("rwg2probe")
                .info("rwg2probe runs {} world generators in a fixed order", FixedGenerators.install());
    }

    @EventHandler
    public void serverStarted(FMLServerStartedEvent event) throws IOException {
        TickQueue queue = new TickQueue();
        FMLCommonHandler.instance().bus().register(queue);
        ProbeServer.start(
                Integer.getInteger("rwg2probe.port", 0),
                new File(PORT_FILE),
                new Ops(Token.read(new File(TOKEN_FILE)), queue));
        Lifecycle.watchIdle(Double.parseDouble(System.getProperty("rwg2probe.idleMinutes", "5")));
    }
}
