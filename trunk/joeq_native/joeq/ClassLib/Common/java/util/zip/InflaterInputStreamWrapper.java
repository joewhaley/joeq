/*
 * InflaterInputStreamWrapper.java
 *
 * @author  John Whaley
 * @version 
 */

package ClassLib.Common.java.util.zip;

import Run_Time.Reflection;
import Bootstrap.PrimordialClassLoader;
import Clazz.jq_Class;
import Clazz.jq_InstanceMethod;
import jq;

public class InflaterInputStreamWrapper extends java.util.zip.InflaterInputStream {
    private boolean isClosed;
    private boolean eof;
    private final ZipFile zf;
    public InflaterInputStreamWrapper(ZipFile zf, java.io.InputStream in, java.util.zip.Inflater inflater) {
        super(in, inflater);
        this.zf = zf;
        this.isClosed = false; this.eof = false;
    }
    public void close() throws java.io.IOException {
        if (!this.isClosed) {
            zf.releaseInflater0(inf);
            in.close();
            isClosed = true;
        }
    }
    protected void fill() throws java.io.IOException {
        if (eof) throw new java.io.EOFException("Unexpected end of ZLIB input stream");
        len = this.in.read(buf, 0, buf.length);
        if (len == -1) {
            buf[0] = 0;
            len = 1;
            eof = true;
        }
        inf.setInput(buf, 0, len);
    }
    public int available() throws java.io.IOException {
        if (super.available() != 0) return this.in.available();
        return 0;
    }
    
}
