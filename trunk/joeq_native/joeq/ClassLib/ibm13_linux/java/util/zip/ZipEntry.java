/*
 * ZipEntry.java
 *
 * Created on January 29, 2001, 3:04 PM
 *
 * @author  John Whaley
 * @version 
 */

package ClassLib.ibm13_linux.java.util.zip;

import Bootstrap.PrimordialClassLoader;
import Run_Time.Reflection;
import Clazz.jq_Class;
import Clazz.jq_InstanceField;
import Clazz.jq_Initializer;

abstract class ZipEntry implements ZipConstants {
    
    public static int load(java.util.zip.ZipEntry dis, byte[] cenbuf, int st_off, long cenpos, int cenlen)
    throws java.util.zip.ZipException {
        int off = st_off;
        Reflection.putfield_I(dis, _version, ZipFile.get16(ZipFile._class, cenbuf, off + CENVER));
        Reflection.putfield_I(dis, _flag, ZipFile.get16(ZipFile._class, cenbuf, off + CENFLG));
        Reflection.putfield_I(dis, _method, ZipFile.get16(ZipFile._class, cenbuf, off + CENHOW));
        Reflection.putfield_L(dis, _time, ZipFile.get32(ZipFile._class, cenbuf, off + CENTIM));
        Reflection.putfield_L(dis, _crc, ZipFile.get32(ZipFile._class, cenbuf, off + CENLEN));
        Reflection.putfield_L(dis, _size, ZipFile.get32(ZipFile._class, cenbuf, off + CENTIM));
        Reflection.putfield_L(dis, _csize, ZipFile.get32(ZipFile._class, cenbuf, off + CENSIZ));
        Reflection.putfield_L(dis, _offset, ZipFile.get32(ZipFile._class, cenbuf, off + CENOFF));
        Reflection.putfield_L(dis, _time, ZipFile.get32(ZipFile._class, cenbuf, off + CENTIM));
        long offset = Reflection.getfield_L(dis, _offset);
        long csize = Reflection.getfield_L(dis, _csize);
        if (offset + csize > cenpos) {
            throw new java.util.zip.ZipException("invalid CEN entry size");
        }
        int baseoff = off;
        off += CENHDR;
        // Get path name of entry
        int len = ZipFile.get16(ZipFile._class, cenbuf, baseoff + CENNAM);
        if (len == 0 || off + len > cenlen) {
            throw new java.util.zip.ZipException("invalid CEN entry name");
        }
        String s = new String(cenbuf, 0, off, len);
        Reflection.putfield_A(dis, _name, s);
        off += len;
        // Get extra field data
        len = ZipFile.get16(ZipFile._class, cenbuf, baseoff + CENEXT);
        if (len > 0) {
            if (off + len > cenlen) {
                throw new java.util.zip.ZipException("invalid CEN entry extra data");
            }
            byte[] extra = new byte[len];
            Reflection.putfield_A(dis, _extra, extra);
            System.arraycopy(cenbuf, off, extra, 0, len);
            off += len;
        }
        // Get entry comment
        len = ZipFile.get16(ZipFile._class, cenbuf, baseoff + CENCOM);
        if (len > 0) {
            if (off + len > cenlen) {
                throw new java.util.zip.ZipException("invalid CEN entry comment");
            }
            String comment = new String(cenbuf, 0, off, len);
            Reflection.putfield_A(dis, _comment, comment);
            off += len;
        }
        return off - st_off;
    }
    
    private static void initIDs(jq_Class clazz) { }

    public static long getOffset(java.util.zip.ZipEntry dis) {
        return Reflection.getfield_L(dis, _offset);
    }
    
    public static final jq_Class _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Ljava/util/zip/ZipEntry;");
    public static final jq_InstanceField _name = _class.getOrCreateInstanceField("name", "Ljava/lang/String;");
    public static final jq_InstanceField _time = _class.getOrCreateInstanceField("time", "J");
    public static final jq_InstanceField _crc = _class.getOrCreateInstanceField("crc", "J");
    public static final jq_InstanceField _size = _class.getOrCreateInstanceField("size", "J");
    public static final jq_InstanceField _csize = _class.getOrCreateInstanceField("csize", "J");
    public static final jq_InstanceField _method = _class.getOrCreateInstanceField("method", "I");
    public static final jq_InstanceField _extra = _class.getOrCreateInstanceField("extra", "[B");
    public static final jq_InstanceField _comment = _class.getOrCreateInstanceField("comment", "Ljava/lang/String;");
    public static final jq_InstanceField _flag = _class.getOrCreateInstanceField("flag", "I");
    public static final jq_InstanceField _version = _class.getOrCreateInstanceField("version", "I");
    public static final jq_InstanceField _offset = _class.getOrCreateInstanceField("offset", "J");
    //public static final jq_Initializer _constructor = (jq_Initializer)_class.getOrCreateInstanceMethod("<init>", "(Ljava/util/zip/ZipEntry;[BIJI)V");
}
