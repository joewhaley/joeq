/*
 * ZipFile.java
 *
 * Created on July 3, 2002, 3:15 PM
 */

package ClassLib.sun142_linux.java.util.zip;

/**
 *
 * @author  John Whaley
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
