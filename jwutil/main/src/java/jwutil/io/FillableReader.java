// FillableReader.java, created Oct 5, 2004 10:35:52 PM 2004 by jwhaley
// Copyright (C) 2004 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package jwutil.io;

import java.io.IOException;
import java.io.Reader;
import java.io.Writer;

/**
 * FillableReader
 * 
 * @author jwhaley
 * @version $Id$
 */
public class FillableReader extends Reader {

    public Writer getWriter() {
        FISWriter os = new FISWriter();
        return os;
    }

    public class FISWriter extends Writer {

        public void write(char c) {
            FillableReader.this.write(c);
        }
        
        /* (non-Javadoc)
         * @see java.io.Writer#write(char[], int, int)
         */
        public void write(char[] cbuf, int off, int len) throws IOException {
            FillableReader.this.write(cbuf, off, len);
        }

        /* (non-Javadoc)
         * @see java.io.Writer#flush()
         */
        public void flush() throws IOException {
        }

        /* (non-Javadoc)
         * @see java.io.Writer#close()
         */
        public void close() throws IOException {
            FillableReader.this.close();
        }
        
    }
    
    char[] buffer;
    volatile int start, end;
    
    public FillableReader() {
        buffer = new char[512];
        start = end = 0;
    }
    
    /* (non-Javadoc)
     * @see java.io.Reader#ready()
     */
    public boolean ready() {
        return start != end;
    }
    
    /* (non-Javadoc)
     * @see java.io.Reader#read()
     */
    public int read() throws IOException {
        synchronized (lock) {
            while (start == end) {
                try {
                    lock.wait();
                } catch (InterruptedException x) { }
            }
            int res = buffer[start++];
            if (start == buffer.length) start = 0;
            lock.notify();
            return res;
        }
    }

    /* (non-Javadoc)
     * @see java.io.Reader#read(char[], int, int)
     */
    public int read(char[] b, int off, int len) {
        synchronized (lock) {
            while (start == end) {
                try {
                    lock.wait();
                } catch (InterruptedException x) { }
            }
            int a = Math.min(len, (start < end ? end : buffer.length) - start);
            System.arraycopy(buffer, start, b, off, a);
            start += a;
            if (start == buffer.length) start = 0;
            lock.notify();
            return a;
        }
    }
    
    private int nextEnd(int count) {
        int newEnd = end + count;
        if (newEnd >= buffer.length) newEnd -= buffer.length;
        return newEnd;
    }
    
    public void write(int b) {
        synchronized (lock) {
            while (start == nextEnd(1)) {
                try {
                    lock.wait();
                } catch (InterruptedException x) { }
            }
            buffer[end++] = (char) b;
            if (end == buffer.length) end = 0;
            lock.notify();
        }
    }
    
    public void write(char[] b) {
        write(b, 0, b.length);
    }
    
    public void write(char[] b, int off, int len) {
        synchronized (lock) {
            while (len > 0) {
                int upTo, a;
                for (;;) {
                    upTo = start <= end ? (start > 0 ? buffer.length : buffer.length - 1) : start - 1;
                    a = upTo - end;
                    if (a > 0) break;
                    try {
                        lock.wait();
                    } catch (InterruptedException x) { }
                }
                a = Math.min(a, len);
                _write(b, off, a);
                lock.notify();
                off += a;
                len -= a;
            }
        }
    }
    
    private void _write(char[] b, int off, int len) {
        System.arraycopy(b, off, buffer, end, len);
        end += len;
        if (end == buffer.length) end = 0;
    }

    public void write(String s) {
        int len = s.length();
        synchronized (lock) {
            for (int i = 0 ; i < len ; i++) {
                write(s.charAt(i));
            }
        }
    }

    /* (non-Javadoc)
     * @see java.io.Reader#close()
     */
    public void close() throws IOException {
    }

}
