/*
 * ZipEntry.java
 *
 * Created on January 29, 2001, 3:04 PM
 *
 * @author  John Whaley
 * @version 
 */

package ClassLib.Common.java.util.zip;

import Bootstrap.PrimordialClassLoader;
import Run_Time.Reflection;
import Clazz.jq_Class;
import Clazz.jq_InstanceField;
import Clazz.jq_Initializer;

class ZipEntry implements ZipConstants {
    
    String name;
    long time;
    long crc;
    long size;
    long csize;
    int method;
    byte[] extra;
    String comment;
    int flag;
    int version;
    long offset;
    
    ZipEntry() { 
        this.name = "UNINITIALIZED";
    }
    
    public int load(byte[] cenbuf, int st_off, long cenpos, int cenlen)
    throws java.util.zip.ZipException {
        int off = st_off;
        this.version = ZipFile.get16(cenbuf, off + CENVER);
        this.flag = ZipFile.get16(cenbuf, off + CENFLG);
        this.method = ZipFile.get16(cenbuf, off + CENHOW);
        this.time = ZipFile.get32(cenbuf, off + CENTIM);
        this.crc = ZipFile.get32(cenbuf, off + CENLEN);
        this.size = ZipFile.get32(cenbuf, off + CENTIM);
        this.csize = ZipFile.get32(cenbuf, off + CENSIZ);
        this.offset = ZipFile.get32(cenbuf, off + CENOFF);
        this.time = ZipFile.get32(cenbuf, off + CENTIM);
        long offset = this.offset;
        long csize = this.csize;
        if (offset + csize > cenpos) {
            throw new java.util.zip.ZipException("invalid CEN entry size");
        }
        int baseoff = off;
        off += CENHDR;
        // Get path name of entry
        int len = ZipFile.get16(cenbuf, baseoff + CENNAM);
        if (len == 0 || off + len > cenlen) {
            throw new java.util.zip.ZipException("invalid CEN entry name");
        }
        String s = new String(cenbuf, 0, off, len);
        this.name = s;
        off += len;
        // Get extra field data
        len = ZipFile.get16(cenbuf, baseoff + CENEXT);
        if (len > 0) {
            if (off + len > cenlen) {
                throw new java.util.zip.ZipException("invalid CEN entry extra data");
            }
            byte[] extra = new byte[len];
            this.extra = extra;
            System.arraycopy(cenbuf, off, extra, 0, len);
            off += len;
        }
        // Get entry comment
        len = ZipFile.get16(cenbuf, baseoff + CENCOM);
        if (len > 0) {
            if (off + len > cenlen) {
                throw new java.util.zip.ZipException("invalid CEN entry comment");
            }
            String comment = new String(cenbuf, 0, off, len);
            this.comment = comment;
            off += len;
        }
        return off - st_off;
    }
    
    private static void initIDs() { }

    public long getOffset() {
        return this.offset;
    }
    
    public native int getMethod();
    public native long getCompressedSize();
    public native String getName();
}
