/*
 * System.java
 *
 * Created on July 3, 2002, 2:44 PM
 */

package ClassLib.sun13_win32.java.lang;
import Bootstrap.PrimordialClassLoader;

/**
 *
 * @author  John Whaley
 * @version 
 */
public class System {

    private static java.util.Properties initProperties(java.util.Properties props) {
        props.setProperty("java.version", "1.3.0");
        props.setProperty("java.vendor", "joeq");
        props.setProperty("java.vendor.url", "http://www.joewhaley.com");
        props.setProperty("java.class.version", "47.0");
        
        // TODO: read these properties from environment.
        props.setProperty("java.home", "G:\\jdk1.3\\jre");
        props.setProperty("os.name", "Windows 2000");
        props.setProperty("os.arch", "x86");
        props.setProperty("os.version", "5.0");
        props.setProperty("file.separator", "\\");
        props.setProperty("path.separator", ";");
        props.setProperty("line.separator", "\r\n");
        props.setProperty("user.name", "jwhaley");
        props.setProperty("user.home", "G:\\Documents and Settings\\John Whaley");
        props.setProperty("user.dir", "G:\\joeq");
        props.setProperty("java.class.path", PrimordialClassLoader.loader.classpathToString());
        return props;
    }

}
