// ZipFile.java, created Thu May  8 13:39:30 2003 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package ClassLib.sun142_linux.java.util.zip;

/**
 * ZipFile
 *
 * @author  John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public abstract class ZipFile {

    private String name;
    private java.util.Vector inflaters;
    private java.io.RandomAccessFile raf;
    private java.util.Hashtable entries;
    
    public void __init__(String name) throws java.io.IOException {
        this.name = name;
        java.io.RandomAccessFile raf = new java.io.RandomAccessFile(name, "r");
        this.raf = raf;
        this.inflaters = new java.util.Vector();
        this.readCEN();
    }
    
    private native void readCEN() throws java.io.IOException;
}
