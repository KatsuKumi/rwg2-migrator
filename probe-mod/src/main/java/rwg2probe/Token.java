package rwg2probe;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.security.MessageDigest;

/** The session secret the tool writes into a user-only file before the boot; it is never on the command line. */
final class Token {

    private final byte[] secret;

    private Token(byte[] secret) {
        this.secret = secret;
    }

    static Token read(File file) throws IOException {
        if (!file.isFile()) return new Token(null);
        return new Token(
                new String(Files.readAllBytes(file.toPath()), StandardCharsets.UTF_8).trim()
                        .getBytes(StandardCharsets.UTF_8));
    }

    void check(String given) {
        if (secret == null || secret.length == 0)
            throw new SecurityException("the probe has no token file; restart it from the tool");
        if (given == null || !MessageDigest.isEqual(secret, given.getBytes(StandardCharsets.UTF_8)))
            throw new SecurityException("missing or wrong token");
    }
}
