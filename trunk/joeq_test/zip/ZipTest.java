// ZipTest.java, created Aug 9, 2003 12:54:02 AM by John Whaley
// Copyright (C) 2003 John Whaley
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package zip;

import java.io.*;
import java.util.*;

/**
 * ZipTest
 * 
 * @author John Whaley
 * @version $Id$
 */
public class ZipTest {

    public static void main(String[] args) throws Exception {
        java.util.zip.ZipFile f = new java.util.zip.ZipFile(args[0]);
        Enumeration e = f.entries();
        int num = 0;
        while (e.hasMoreElements()) {
            java.util.zip.ZipEntry ze = (java.util.zip.ZipEntry) e.nextElement();
            String name = ze.getName();
            if (name.endsWith(".class")) {
                DataInput i = new DataInputStream(f.getInputStream(ze));
                int val = i.readInt();
                if (val != 0xcafebabe) {
                    System.out.println(ze+" = "+Integer.toHexString(val));
                }
                ++num;
                try {
                    Class.forName(name.substring(0, name.length()-6).replace('/','.'));
                } catch (ClassFormatError x) {
                    System.out.println(name+": "+x);
                    Debugger.OnlineDebugger.debuggerEntryPoint();
                } catch (UnsatisfiedLinkError x) {
                    System.out.println(name+": "+x);
                }
            }
        }
        System.out.println(num+" classes");
    }

}
