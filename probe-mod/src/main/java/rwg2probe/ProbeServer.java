package rwg2probe;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.Arrays;
import java.util.Base64;
import java.util.HashSet;
import java.util.Set;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import com.google.gson.Gson;
import com.google.gson.JsonObject;

/**
 * One JSON request per line, one JSON reply per line; requests that use RWG2's static offsets or the world run one at a
 * time.
 */
final class ProbeServer implements Runnable {

    private static final Logger LOG = LogManager.getLogger("rwg2probe");
    private static final Gson GSON = new Gson();
    private static final int BACKLOG = 8;
    private static final Set<String> CONCURRENT = new HashSet<>(Arrays.asList("status", "stop", "biomes"));

    private final ServerSocket socket;
    private final Ops ops;
    private final Object exclusive = new Object();

    private ProbeServer(ServerSocket socket, Ops ops) {
        this.socket = socket;
        this.ops = ops;
    }

    static void start(int port, File portFile, Ops ops) throws IOException {
        ServerSocket socket = new ServerSocket(port, BACKLOG, InetAddress.getLoopbackAddress());
        File tmp = new File(portFile.getPath() + ".tmp");
        Files.write(tmp.toPath(), Integer.toString(socket.getLocalPort()).getBytes(StandardCharsets.US_ASCII));
        Files.move(tmp.toPath(), portFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
        Thread thread = new Thread(new ProbeServer(socket, ops), "rwg2probe");
        thread.setDaemon(true);
        thread.start();
        LOG.info("rwg2probe listening on 127.0.0.1:{}", socket.getLocalPort());
    }

    @Override
    public void run() {
        while (!socket.isClosed()) {
            try {
                Socket client = socket.accept();
                Thread thread = new Thread(() -> serve(client), "rwg2probe-client");
                thread.setDaemon(true);
                thread.start();
            } catch (IOException e) {
                LOG.warn("rwg2probe accept failed: {}", e.toString());
            }
        }
    }

    private void serve(Socket socket) {
        try (Socket client = socket) {
            BufferedReader in = new BufferedReader(
                    new InputStreamReader(client.getInputStream(), StandardCharsets.UTF_8));
            OutputStream out = client.getOutputStream();
            for (String line; (line = in.readLine()) != null;) {
                if (line.trim().isEmpty()) continue;
                Lifecycle.begin();
                try {
                    handle(line, out);
                    out.flush();
                } finally {
                    Lifecycle.end();
                }
            }
        } catch (IOException e) {
            LOG.warn("rwg2probe connection ended: {}", e.toString());
        }
    }

    private byte[] run(String op, JsonObject request, JsonObject reply) throws Exception {
        if (CONCURRENT.contains(op)) return ops.run(op, request, reply);
        synchronized (exclusive) {
            return ops.run(op, request, reply);
        }
    }

    private void handle(String line, OutputStream out) throws IOException {
        long started = System.nanoTime();
        JsonObject reply = new JsonObject();
        byte[] data = null;
        try {
            JsonObject request = GSON.fromJson(line, JsonObject.class);
            String op = request.get("op").getAsString();
            reply.addProperty("op", op);
            data = run(op, request, reply);
            reply.addProperty("ok", true);
        } catch (Throwable e) {
            if (!(e instanceof SecurityException)) LOG.warn("rwg2probe request failed", e);
            reply.addProperty("ok", false);
            reply.addProperty("error", e.toString());
            reply.remove("stopping");
            data = null;
        }
        reply.addProperty("seconds", (System.nanoTime() - started) / 1e9);
        write(out, reply, data);
        if (reply.has("stopping")) Lifecycle.shutdown("the tool asked it to stop");
    }

    private static void write(OutputStream out, JsonObject reply, byte[] data) throws IOException {
        String head = GSON.toJson(reply);
        if (data == null) {
            out.write((head + "\n").getBytes(StandardCharsets.UTF_8));
            return;
        }
        out.write((head.substring(0, head.length() - 1) + ",\"data\":\"").getBytes(StandardCharsets.UTF_8));
        out.write(Base64.getEncoder().encode(data));
        out.write("\"}\n".getBytes(StandardCharsets.UTF_8));
    }
}
