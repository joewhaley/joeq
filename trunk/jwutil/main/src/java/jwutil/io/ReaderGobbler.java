// ReaderGobbler.java, created Oct 5, 2004 9:27:40 PM 2004 by jwhaley
// Copyright (C) 2004 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package jwutil.io;

import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.Reader;
import java.io.Writer;

/**
 * ReaderGobbler
 * 
 * @author jwhaley
 * @version $Id$
 */
public class ReaderGobbler extends Thread {
    protected Reader is;
    protected Writer out;
    protected Output out2;
    
    /**
     * A simple interface clients can implement to receive output from an reader.
     * 
     * @author jwhaley
     * @version $Id$
     */
    public static interface Output {
        void write(char[] buffer, int off, int len);
    }
    
    public ReaderGobbler(Reader is) {
        this(is, new OutputStreamWriter(System.out));
    }
    
    public ReaderGobbler(Writer o) {
        this(new InputStreamReader(System.in), o);
    }
    
    public ReaderGobbler(Output o) {
        this(new InputStreamReader(System.in), o);
    }
    
    public ReaderGobbler(Reader is, Writer o) {
        this.is = is;
        this.out = o;
    }
    
    public ReaderGobbler(Reader is, Output o) {
        this.is = is;
        this.out2 = o;
    }
    
    public void setReader(Reader r) {
        this.is = r;
    }
    
    public void setWriter(Writer o) {
        this.out = o;
        this.out2 = null;
    }
    
    public void setOutput(Output o) {
        this.out = null;
        this.out2 = o;
    }
    
    public void run() {
        try {
            char[] buffer = new char[512];
            boolean eos = false;
            while (!eos) {
                int i = 0;
                // while chars are available, read them up to buffer.length chars.
                while (i < buffer.length && is.ready()) {
                    int r = is.read();
                    if (r < 0) {
                        eos = true; break;
                    } else {
                        buffer[i++] = (char) r;
                    }
                }
                // write the chars we just read.
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
