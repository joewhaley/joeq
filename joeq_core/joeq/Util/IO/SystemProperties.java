//SystemProperties.java, created Sun Dec  7 14:20:28 PST 2003
//Copyright (C) 2004 Godmar Back <gback@cs.utah.edu, @stanford.edu>
//Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Util.IO;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.Properties;

/**
 * Read system properties from a file.
 * @version $Id$
 * @author gback
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
