// InputStreamGobbler.java, created Oct 5, 2004 8:44:20 PM 2004 by jwhaley
// Copyright (C) 2004 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package jwutil.io;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

/**
 * InputStreamGobbler is a thread that reads from a given InputStream and writes whatever
 * is read to an OutputStream.
 * 
 * @author John Whaley
 * @version $Id$
 */
public class InputStreamGobbler extends Thread {
    protected InputStream is;
    protected OutputStream out;
    protected Output out2;
    
    /**
     * A simple interface clients can implement to receive output from an input stream.
     * 
     * @author jwhaley
     * @version $Id$
     */
    public static interface Output {
        void write(byte[] buffer, int off, int len);
    }
    
    public InputStreamGobbler(InputStream is) {
        this(is, System.out);
    }
    
    public InputStreamGobbler(OutputStream o) {
        this(System.in, o);
    }
    
    public InputStreamGobbler(Output o) {
        this(System.in, o);
    }
    
    public InputStreamGobbler(InputStream is, OutputStream o) {
        this.is = is;
        this.out = o;
    }
    
    public InputStreamGobbler(InputStream is, Output o) {
        this.is = is;
        this.out2 = o;
    }
    
    public void setInput(InputStream r) {
        this.is = r;
    }
    
    public void setOutput(OutputStream o) {
        this.out = o;
        this.out2 = null;
    }
    
    public void setOutput(Output o) {
        this.out = null;
        this.out2 = o;
    }
    
    public void run() {
        try {
            byte[] buffer = new byte[1024];
            boolean eos = false;
            while (!eos) {
                int i;
                // if there is nothing available, just block on reading a byte.
                if (is.available() == 0) {
                    int r = is.read();
                    if (r < 0) {
                        eos = true;
                        break;
                    } else {
                        buffer[0] = (byte) r;
                        i = 1;
                    }
                } else {
                    i = 0;
                    // while bytes are available, read them up to buffer.length bytes.
                    while (i < buffer.length && is.available() > 0) {
                        int r = is.read(buffer, i, Math.min(is.available(), buffer.length-i));
                        if (r < 0) {
                            eos = true; break;
                        } else if (r == 0) {
                            break;
                        } else {
                            i += r;
                        }
                    }
                }
                // write the bytes we just read.
                if (out != null) {
                    out.write(buffer, 0, i);
                } else {
                    out2.write(buffer, 0, i);
                }
            }
        } catch (IOException ioe) {
            ioe.printStackTrace();  
        }
    }
}
