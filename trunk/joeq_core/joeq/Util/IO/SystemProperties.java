package joeq.Util.IO;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Properties;

/**
 * Read system properties from a file.
 */
public class SystemProperties {
    public static void read(String filename) {
        try {
            FileInputStream propFile = new FileInputStream(filename);
            Properties p = new Properties(System.getProperties());
            p.load(propFile);
            System.setProperties(p);
        } catch (IOException ie) {
            ;   // silent
        }
    }
}
