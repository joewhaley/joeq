/*
 * ZipFile.java
 *
 * Created on January 29, 2001, 3:20 PM
 *
 * @author  John Whaley
 * @version 
 */

package ClassLib.ibm13_linux.java.util.zip;

import Bootstrap.PrimordialClassLoader;
import java.io.RandomAccessFile;
import java.util.Hashtable;
import java.util.Enumeration;
import jq;
import Clazz.jq_Class;
import Clazz.jq_InstanceField;
import Clazz.jq_InstanceMethod;
import Run_Time.Reflection;

public abstract class ZipFile implements ZipConstants {
    public static final boolean TRACE = false;
    
    private RandomAccessFile raf;
    private Hashtable entries;
    private long cenpos;
    private long pos;

    private static void initIDs(jq_Class clazz) { }
    public static void __init__(java.util.zip.ZipFile dis, String name) throws java.io.IOException {
	Reflection.putfield_A(dis, _name, name); // must be before putfield of raf
        RandomAccessFile raf = new RandomAccessFile(name, "r");
	Reflection.putfield_A(dis, _raf, raf);
	readCEN(dis);
    }
    public static void __init__(java.util.zip.ZipFile dis, java.io.File file, int mode) throws java.io.IOException {
        // delete mode not yet supported.
        jq.assert(mode == java.util.zip.ZipFile.OPEN_READ);
        __init__(dis, file.getPath());
    }
    public static java.util.zip.ZipEntry getEntry(java.util.zip.ZipFile dis, String name) {
        if (TRACE) System.out.println(dis+": getting entry "+name);
        Hashtable entries = (Hashtable)Reflection.getfield_A(dis, _entries);
        return (java.util.zip.ZipEntry)entries.get(name);
    }
    public static Enumeration entries(java.util.zip.ZipFile dis) {
        Hashtable entries = (Hashtable)Reflection.getfield_A(dis, _entries);
	return entries.elements();
    }
    public static int size(java.util.zip.ZipFile dis) {
        Hashtable entries = (Hashtable)Reflection.getfield_A(dis, _entries);
        if (TRACE) System.out.println(dis+": getting size = "+entries.size());
	return entries.size();
    }
    public static void close(java.util.zip.ZipFile dis) throws java.io.IOException {
        if (TRACE) System.out.println(dis+": closing file");
        RandomAccessFile raf = (RandomAccessFile)Reflection.getfield_A(dis, _raf);
	if (raf != null) {
	    raf.close();
	    raf = null;
	}
    }
    public static java.io.InputStream getInputStream(final java.util.zip.ZipFile dis, java.util.zip.ZipEntry ze)
    throws java.io.IOException {
        if (ze == null) return null;
	java.io.InputStream in = (java.io.InputStream)ZipFileInputStream._class.newInstance();
        ZipFileInputStream.__init__(in, dis, ze);
        if (TRACE) System.out.println(dis+": getting input stream for "+ze+" = "+in);
	switch (ze.getMethod()) {
	case java.util.zip.ZipEntry.STORED:
	    return in;
	case java.util.zip.ZipEntry.DEFLATED:
            java.util.zip.Inflater inflater;
            try {
                // invoke private method
                inflater = (java.util.zip.Inflater)Reflection.invokeinstance_A(_getInflater, dis);
            } catch (Error x) {
                throw x;
            } catch (Throwable x) {
                jq.UNREACHABLE(); return null;
            }
            if (TRACE) System.out.println(dis+": using inflater "+inflater);
            // Overridden InflaterInputStream to add a zero byte at the end of the stream.
            return new InflaterInputStreamWrapper(dis, in, inflater);
	default:
	    throw new java.util.zip.ZipException("invalid compression method");
	}
    }
    
    // Overridden InflaterInputStream to add a zero byte at the end of the stream.
    static class InflaterInputStreamWrapper extends java.util.zip.InflaterInputStream {
        private boolean isClosed;
        private boolean eof;
        private final java.util.zip.ZipFile dis;
        InflaterInputStreamWrapper(java.util.zip.ZipFile dis, java.io.InputStream in, java.util.zip.Inflater inflater) {
            super(in, inflater);
            this.dis = dis;
            isClosed = false; eof = false;
        }
        public void close() throws java.io.IOException {
            if (!isClosed) {
                try {
                    // invoke private method
                    Reflection.invokeinstance_A(_releaseInflater, dis, inf);
                } catch (Error x) {
                    throw x;
                } catch (Throwable x) {
                    jq.UNREACHABLE();
                }
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
    
    private abstract static class ZipFileInputStream extends java.io.InputStream {
        private java.util.zip.ZipFile zf;
        private java.util.zip.ZipEntry ze;
        private long pos;
        private long count;
        
        public static void __init__(Object/*java.util.zip.ZipFile.ZipFileInputStream*/ dis,
                                    java.util.zip.ZipFile zf, java.util.zip.ZipEntry ze)
        throws java.io.IOException {
            Reflection.putfield_A(dis, _zf, zf);
            Reflection.putfield_A(dis, _ze, ze);
            readLOC(dis);
        }
        
	public static int read(Object/*java.util.zip.ZipFile.ZipFileInputStream*/ dis, byte b[], int off, int len) throws java.io.IOException {
            long count = Reflection.getfield_L(dis, _count);
            if (TRACE) System.out.println(dis+": reading off="+off+" len="+len);
            if (count == 0) {
                return -1;
            }
            if (len > count) {
                len = (int)Math.min(count, Integer.MAX_VALUE);
            }
            long pos = Reflection.getfield_L(dis, _pos);
            java.util.zip.ZipFile zf = (java.util.zip.ZipFile)Reflection.getfield_A(dis, _zf);
            len = ZipFile.read(zf, pos, b, off, len);
            if (len == -1) {
                throw new java.util.zip.ZipException("premature EOF");
            }
            Reflection.putfield_L(dis, _pos, pos+len);
            Reflection.putfield_L(dis, _count, count-len);
            return len;
	}

	public static int read(Object/*java.util.zip.ZipFile.ZipFileInputStream*/ dis) throws java.io.IOException {
            long count = Reflection.getfield_L(dis, _count);
            if (count == 0) {
                return -1;
            }
            java.util.zip.ZipFile zf = (java.util.zip.ZipFile)Reflection.getfield_A(dis, _zf);
            long pos = Reflection.getfield_L(dis, _pos);
            if (TRACE) System.out.println(dis+": reading pos="+pos);
            int n = ZipFile.read(zf, pos);
            if (n == -1) {
                throw new java.util.zip.ZipException("premature EOF");
            }
            Reflection.putfield_L(dis, _pos, pos+1);
            Reflection.putfield_L(dis, _count, count-1);
            if (TRACE) System.out.println(dis+": new pos="+(pos+1));
            if (TRACE) System.out.println(dis+": new count="+(count-1));
            return n;
	}

	public static long skip(Object/*java.util.zip.ZipFile.ZipFileInputStream*/ dis, long n) {
            long count = Reflection.getfield_L(dis, _count);
            if (n > count) {
                n = count;
            }
            if (TRACE) System.out.println(dis+": skipping "+n);
            long pos = Reflection.getfield_L(dis, _pos);
            Reflection.putfield_L(dis, _pos, pos+n);
            Reflection.putfield_L(dis, _count, count-n);
            if (TRACE) System.out.println(dis+": new pos="+(pos+n));
            if (TRACE) System.out.println(dis+": new count="+(count-n));
            return n;
	}

        public static int available(Object/*java.util.zip.ZipFile.ZipFileInputStream*/ dis) {
            long count = Reflection.getfield_L(dis, _count);
            return (int)Math.min(count, Integer.MAX_VALUE);
	}

        private static void readLOC(Object/*java.util.zip.ZipFile.ZipFileInputStream*/ dis) throws java.io.IOException {
            // Read LOC header and check signature
            byte locbuf[] = new byte[LOCHDR];
            java.util.zip.ZipFile zf = (java.util.zip.ZipFile)Reflection.getfield_A(dis, _zf);
            java.util.zip.ZipEntry ze = (java.util.zip.ZipEntry)Reflection.getfield_A(dis, _ze);
            long offset = ZipEntry.getOffset(ze);
            if (TRACE) System.out.println(dis+": reading LOC, offset="+offset);
            ZipFile.read(zf, offset, locbuf, 0, LOCHDR);
            if (ZipFile.get32(_class, locbuf, 0) != LOCSIG) {
                throw new java.util.zip.ZipException("invalid LOC header signature");
            }
            // Get length and position of entry data
            long count = ze.getCompressedSize();
            Reflection.putfield_L(dis, _count, count);
            if (TRACE) System.out.println(dis+": count="+count);
            long pos = ZipEntry.getOffset(ze) + LOCHDR + ZipFile.get16(_class, locbuf, LOCNAM) + ZipFile.get16(_class, locbuf, LOCEXT);
            Reflection.putfield_L(dis, _pos, pos);
            if (TRACE) System.out.println(dis+": pos="+pos);
            long cenpos = Reflection.getfield_L(zf, ZipFile._cenpos);
            if (TRACE) System.out.println(dis+": cenpos="+cenpos);
            if (pos + count > cenpos) {
                throw new java.util.zip.ZipException("invalid LOC header format");
            }
        }
        public static final jq_Class _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Ljava/util/zip/ZipFile$ZipFileInputStream;");
        public static final jq_InstanceField _zf = _class.getOrCreateInstanceField("zf", "Ljava/util/zip/ZipFile;");
        public static final jq_InstanceField _ze = _class.getOrCreateInstanceField("ze", "Ljava/util/zip/ZipEntry;");
        public static final jq_InstanceField _pos = _class.getOrCreateInstanceField("pos", "J");
        public static final jq_InstanceField _count = _class.getOrCreateInstanceField("count", "J");
    }
    private /*synchronized*/ static int read(java.util.zip.ZipFile dis, long pos, byte b[], int off, int len)
	throws java.io.IOException
    {
        if (TRACE) System.out.println(dis+": reading file pos="+pos+" off="+off+" len="+len);
        RandomAccessFile raf = (RandomAccessFile)Reflection.getfield_A(dis, _raf);
        if (pos != Reflection.getfield_L(dis, _pos)) {
	    raf.seek(pos);
	}
	int n = raf.read(b, off, len);
        if (TRACE) System.out.println(dis+": number read="+n);
        if (TRACE) System.out.println(dis+": current pos="+(pos+n));
	if (n > 0) {
            Reflection.putfield_L(dis, _pos, pos+n);
	}
	return n;
    }
    private /*synchronized*/ static int read(java.util.zip.ZipFile dis, long pos) throws java.io.IOException {
        RandomAccessFile raf = (RandomAccessFile)Reflection.getfield_A(dis, _raf);
        if (TRACE) System.out.println(dis+": read pos="+pos);
        if (pos != Reflection.getfield_L(dis, _pos)) {
            if (TRACE) System.out.println(dis+": seeking to "+pos);
	    raf.seek(pos);
	}
	int n = raf.read();
        if (TRACE) System.out.println(dis+": byte read="+n);
	if (n > 0) {
	    Reflection.putfield_L(dis, _pos, pos + 1);
	}
	return n;
    }
    
    private static void readCEN(java.util.zip.ZipFile dis) throws java.io.IOException {
        if (TRACE) System.out.println(dis+": reading CEN...");
	// Find and seek to beginning of END header
	long endpos = findEND(dis);
        if (TRACE) System.out.println(dis+": endpos="+endpos);
	// Read END header and check signature
	byte[] endbuf = new byte[ENDHDR];
        RandomAccessFile raf = (RandomAccessFile)Reflection.getfield_A(dis, _raf);
	raf.readFully(endbuf);
	if (get32(_class, endbuf, 0) != ENDSIG) {
	    throw new java.util.zip.ZipException("invalid END header signature");
	}
	// Get position and length of central directory
        long cenpos = get32(_class, endbuf, ENDOFF);
        if (TRACE) System.out.println(dis+": cenpos="+cenpos);
	Reflection.putfield_L(dis, _cenpos, cenpos);
	int cenlen = (int)get32(_class, endbuf, ENDSIZ);
        if (TRACE) System.out.println(dis+": cenlen="+cenlen);
	if (cenpos + cenlen != endpos) {
	    throw new java.util.zip.ZipException("invalid END header format");
	}
	// Get total number of entries
	int nent = get16(_class, endbuf, ENDTOT);
        if (TRACE) System.out.println(dis+": nent="+nent);
	if (nent * CENHDR > cenlen) {
	    throw new java.util.zip.ZipException("invalid END header format");
	}
	// Check number of drives
	if (get16(_class, endbuf, ENDSUB) != nent) {
	    throw new java.util.zip.ZipException("cannot have more than one drive");
	}
	// Seek to first CEN record and read central directory
	raf.seek(cenpos);
	byte cenbuf[] = new byte[cenlen];
	raf.readFully(cenbuf);
	// Scan entries in central directory and build lookup table.
	Hashtable entries = new Hashtable(nent);
	Reflection.putfield_A(dis, _entries, entries);
	for (int off = 0; off < cenlen; ) {
	    // Check CEN header signature
	    if (get32(_class, cenbuf, off) != CENSIG) {
		throw new java.util.zip.ZipException("invalid CEN header signature");
	    }
            java.util.zip.ZipEntry e;
            if (jq.Bootstrapping) e = new java.util.zip.ZipEntry("bogus");
            else e = (java.util.zip.ZipEntry)ClassLib.ibm13_linux.java.util.zip.ZipEntry._class.newInstance();
            int entrysize = ZipEntry.load(e, cenbuf, off, cenpos, cenlen);
            off += entrysize;
            if (TRACE) System.out.println(dis+": entrysize="+entrysize+" offset="+off);
	    // Add entry to the hash table of entries
            String name = e.getName();
	    entries.put(name, e);
	}
        if (false) { // zip files can have duplicate entries, so we disable this check.
            // Make sure we got the right number of entries
            if (entries.size() != nent) {
                throw new java.util.zip.ZipException("invalid CEN header format");
            }
        }
    }

    private static final int INBUFSIZ = 64;

    private static long findEND(java.util.zip.ZipFile dis) throws java.io.IOException {
	// Start searching backwards from end of file
        RandomAccessFile raf = (RandomAccessFile)Reflection.getfield_A(dis, _raf);
	long len = raf.length();
        if (TRACE) System.out.println(dis+": findEND len="+len);
	raf.seek(len);
	// Set limit on how far back we need to search. The END header
	// must be located within the last 64K bytes of the raf.
	long markpos = Math.max(0, len - 0xffff);
	// Search backwards INBUFSIZ bytes at a time from end of file
	// stopping when the END header signature has been found. Since
	// the signature may straddle a buffer boundary, we need to stash
	// the first 4-1 bytes of the previous record at the end of
	// the current record so that the search may overlap.
	byte buf[] = new byte[INBUFSIZ + 4];
        long pos = 0L; // Reflection.getfield_L(dis, _pos);
	for (pos = len; pos > markpos; ) {
	    int n = Math.min((int)(pos - markpos), INBUFSIZ);
	    pos -= n;
	    raf.seek(pos);
	    raf.readFully(buf, 0, n);
	    while (--n > 0) {
		if (get32(_class, buf, n) == ENDSIG) {
		    // Could be END header, but we need to make sure that
		    // the record extends to the end of the raf.
		    long endpos = pos + n;
		    if (len - endpos < ENDHDR) {
			continue;
		    }
		    raf.seek(endpos);
		    byte endbuf[] = new byte[ENDHDR];
		    raf.readFully(endbuf);
		    int comlen = get16(_class, endbuf, ENDCOM);
                    if (TRACE) System.out.println(dis+": findEND comlen="+comlen);
		    if (endpos + ENDHDR + comlen != len) {
			continue;
		    }
		    // This is definitely the END record, so position
		    // the file pointer at the header and return.
		    raf.seek(endpos);
                    Reflection.putfield_L(dis, _pos, endpos);
                    if (TRACE) System.out.println(dis+": findEND pos=endpos="+endpos);
		    return endpos;
		}
	    }
	}
	throw new java.util.zip.ZipException("not a ZIP file (END header not found)");
    }

    /*
     * Fetch unsigned 16-bit value from byte array at specified offset.
     * The bytes are assumed to be in Intel (little-endian) byte order.
     */
    static final int get16(jq_Class clazz, byte b[], int off) {
	return (b[off] & 0xff) | ((b[off+1] & 0xff) << 8);
    }

    /*
     * Fetch unsigned 32-bit value from byte array at specified offset.
     * The bytes are assumed to be in Intel (little-endian) byte order.
     */
    static final long get32(jq_Class clazz, byte b[], int off) {
	return get16(clazz, b, off) | ((long)get16(clazz, b, off+2) << 16);
    }
    
    public static final jq_Class _class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Ljava/util/zip/ZipFile;");
    public static final jq_InstanceField _raf = _class.getOrCreateInstanceField("raf", "Ljava/io/RandomAccessFile;");
    public static final jq_InstanceField _name = _class.getOrCreateInstanceField("name", "Ljava/lang/String;");
    public static final jq_InstanceField _entries = _class.getOrCreateInstanceField("entries", "Ljava/util/Hashtable;");
    public static final jq_InstanceField _cenpos = _class.getOrCreateInstanceField("cenpos", "J");
    public static final jq_InstanceField _pos = _class.getOrCreateInstanceField("pos", "J");
    public static final jq_InstanceMethod _getInflater = _class.getOrCreateInstanceMethod("getInflater", "()Ljava/util/zip/Inflater;");
    public static final jq_InstanceMethod _releaseInflater = _class.getOrCreateInstanceMethod("releaseInflater", "(Ljava/util/zip/Inflater;)V");
}
