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
        // TODO: read these properties from environment.
        props.setProperty("java.class.version", "47.0");
        props.setProperty("java.home", "C:\\jdk1.3\\jre");
        props.setProperty("java.runtime.name", "Java(TM) 2 Runtime Environment, Standard Edition");
        props.setProperty("java.runtime.version", "1.3.0");
        props.setProperty("java.specification.name", "Java Platform API Specification");
        props.setProperty("java.specification.vendor", "Sun Microsystems, Inc.");
        props.setProperty("java.specification.version", "1.3");
        props.setProperty("java.vendor", "joeq");
        props.setProperty("java.vendor.url", "http://joeq.sourceforge.net");
        props.setProperty("java.vendor.url.bug", "http://joeq.sourceforge.net");
        props.setProperty("java.version", "1.3.0");
        props.setProperty("java.vm.name", "joeq virtual machine");
        props.setProperty("java.vm.specification.name", "Java Virtual Machine Specification");
        props.setProperty("java.vm.specification.vendor", "Sun Microsystems, Inc.");
        props.setProperty("java.vm.specification.version", "1.0");
        props.setProperty("java.vm.vendor", "joeq");
        props.setProperty("java.vm.version", "1.3.0");
        
        props.setProperty("os.arch", "x86");
        props.setProperty("os.name", "Windows 2000");
        props.setProperty("os.version", "5.0");
        
        props.setProperty("file.encoding", "Cp1252");
        props.setProperty("file.encoding.pkg", "sun.io");
        props.setProperty("file.separator", "\\");
        
        props.setProperty("line.separator", "\r\n");
        
        props.setProperty("path.separator", ";");
        
        props.setProperty("user.dir", "C:\\joeq");
        props.setProperty("user.home", "C:\\Documents and Settings\\John Whaley");
        props.setProperty("user.language", "en");
        props.setProperty("user.name", "jwhaley");
        props.setProperty("user.region", "US");
        props.setProperty("user.timezone", "");

	// must be at end: classpathToString() uses some properties from above.
        props.setProperty("java.class.path", PrimordialClassLoader.loader.classpathToString());

        return props;
    }

}
