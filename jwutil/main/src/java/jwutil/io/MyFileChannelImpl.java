// MyFileChannelImpl.java, created Wed Mar  5  0:26:34 2003 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package jwutil.io;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.MyHeapByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;
import java.nio.channels.NonReadableChannelException;
import java.nio.channels.NonWritableChannelException;
import java.nio.channels.ReadableByteChannel;
import java.nio.channels.WritableByteChannel;
import jwutil.util.Assert;

/**
 * An implementation of FileChannel that is backed by a FileInputStream,
 * FileOutputStream, or RandomAccessFile.  This is useful on virtual machines
 * that do not support the normal FileChannel methods (like JDK 1.3).
 * 
 * Not all functionality has been implemented yet.
 * 
 * @author John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public class MyFileChannelImpl
    extends FileChannel
    implements ReadableByteChannel, WritableByteChannel {

    /**
     * Gets the file channel for the given object.  Assumes the object either
     * has a getChannel() method, or it is a FileInputStream, FileOutputStream,
     * or RandomAccessFile.
     * 
     * @param o  object to get file channel from
     * @return  the file channel
     * @throws IOException  if an IO exception occurred while getting the channel
     */
    public static FileChannel getFileChannel(Object o) throws IOException {
        Class c = o.getClass();
        try {
            Method m = c.getMethod("getChannel", null);
            return (FileChannel) m.invoke(o, null);
        } catch (IllegalAccessException x) {
        } catch (NoSuchMethodException x) {
        } catch (InvocationTargetException x) {
            if (x.getTargetException() instanceof IOException)
                throw (IOException)x.getTargetException();
        }
        if (o instanceof FileInputStream)
            return new MyFileChannelImpl((FileInputStream) o);
        if (o instanceof FileOutputStream)
            return new MyFileChannelImpl((FileOutputStream) o);
        if (o instanceof RandomAccessFile)
            return new MyFileChannelImpl((RandomAccessFile) o);
        Assert.UNREACHABLE(o.getClass().toString());
        return null;
    }

    protected FileInputStream fis;
    protected FileOutputStream fos;
    protected RandomAccessFile raf;
    protected long currentPosition;
    
    private MyFileChannelImpl(RandomAccessFile o) throws IOException {
        this.raf = o;
        this.fis = new FileInputStream(o.getFD());
        this.fos = new FileOutputStream(o.getFD());
    }
    
    private MyFileChannelImpl(FileInputStream o) {
        this.fis = o;
    }

    private MyFileChannelImpl(FileOutputStream o) {
        this.fos = o;
    }
    
    /**
     * @see java.nio.channels.ReadableByteChannel#read(ByteBuffer)
     */
    public int read(ByteBuffer b) throws IOException {
        if (fis != null) {
            if (b instanceof MyHeapByteBuffer) {
                if (!b.hasRemaining()) return 0;
                byte[] ba = ((MyHeapByteBuffer) b).getBackingArray();
                int offset = ((MyHeapByteBuffer) b).getOffset();
                int position = b.position();
                int limit = b.limit();
                Assert._assert(position < limit);
                int r = fis.read(ba, offset+position, limit-position);
                if (r == -1) return -1;
                b.position(position + r);
                currentPosition += r;
                return r;
            }
            int n = 0;
            for (;;) {
                if (!b.hasRemaining()) return n;
                //if (fis.available() == 0) return n;
                int r = fis.read();
                if (r == -1) {
                    // end-of-stream
                    if (n != 0) return n;
                    // return -1 if we started at the end-of-stream
                    return -1;
                }
                b.put((byte)r);
                ++n; ++currentPosition;
            }
        } else {
            throw new NonReadableChannelException();
        }
    }

    /**
     * @see java.nio.channels.ScatteringByteChannel#read(ByteBuffer[], int, int)
     */
    public long read(ByteBuffer[] b, int offset, int length)
        throws IOException {
        int n = 0;
        for (int i=0; i<length; ++i) {
            int k = read(b[offset+i]);
            if (k == -1) {
                if (n != 0) return n;
                else return -1;
            }
            n += k;
        }
        return n;
    }

    /**
     * @see java.nio.channels.WritableByteChannel#write(ByteBuffer)
     */
    public int write(ByteBuffer b) throws IOException {
        if (fos != null) {
            if (b instanceof MyHeapByteBuffer) {
                if (!b.hasRemaining()) return 0;
                byte[] ba = ((MyHeapByteBuffer) b).getBackingArray();
                int offset = ((MyHeapByteBuffer) b).getOffset();
                int position = b.position();
                int limit = b.limit();
                int size = limit - position;
                Assert._assert(size > 0);
                fos.write(ba, offset+position, size);
                b.position(limit);
                currentPosition += size;
                return size;
            }
            int n = 0;
            while (b.hasRemaining()) {
                byte r = b.get();
                fos.write(r);
                ++n; ++currentPosition;
            }
            return n;
        } else {
            throw new NonWritableChannelException();
        }
    }

    /**
     * @see java.nio.channels.GatheringByteChannel#write(ByteBuffer[], int, int)
     */
    public long write(ByteBuffer[] b, int offset, int length)
        throws IOException {
        int n = 0;
        for (int i=0; i<length; ++i) {
            n += write(b[offset+i]);
        }
        return n;
    }

    /**
     * @see java.nio.channels.FileChannel#position()
     */
    public long position() throws IOException {
        return currentPosition;
    }

    /**
     * @see java.nio.channels.FileChannel#position(long)
     */
    public FileChannel position(long arg0) throws IOException {
        if (raf == null) {
            // TODO: support seeking on non-RandomAccessFiles
            throw new IOException();
        }
        raf.seek(arg0);
        return this;
    }

    /**
     * @see java.nio.channels.FileChannel#size()
     */
    public long size() throws IOException {
        if (raf == null) {
            // TODO: support size on non-RandomAccessFiles
            throw new IOException();
        }
        return raf.length();
    }

    /**
     * @see java.nio.channels.FileChannel#truncate(long)
     */
    public FileChannel truncate(long arg0) throws IOException {
        if (raf == null) {
            // TODO: support size on non-RandomAccessFiles
            throw new IOException();
        }
        raf.setLength(arg0);
        return this;
    }

    /**
     * @see java.nio.channels.FileChannel#force(boolean)
     */
    public void force(boolean arg0) throws IOException {
        // TODO.
        if (fos != null) {
            fos.flush();
        }
    }

    /**
     * @see java.nio.channels.FileChannel#read(ByteBuffer, long)
     */
    public int read(ByteBuffer arg0, long arg1) throws IOException {
        throw new IOException("not yet implemented");
    }

    /**
     * @see java.nio.channels.FileChannel#write(ByteBuffer, long)
     */
    public int write(ByteBuffer arg0, long arg1) throws IOException {
        throw new IOException("not yet implemented");
    }

    /**
     * @see java.nio.channels.spi.AbstractInterruptibleChannel#implCloseChannel()
     */
    protected void implCloseChannel() throws IOException {
    }

    /**
     * @see java.nio.channels.FileChannel#transferFrom(ReadableByteChannel, long, long)
     */
    public long transferFrom(ReadableByteChannel arg0, long arg1, long arg2)
        throws IOException {
        throw new IOException("not yet implemented");
    }

    /**
     * @see java.nio.channels.FileChannel#transferTo(long, long, WritableByteChannel)
     */
    public long transferTo(long arg0, long arg1, WritableByteChannel arg2)
        throws IOException {
        throw new IOException("not yet implemented");
    }

    /**
     * @see java.nio.channels.FileChannel#lock(long, long, boolean)
     */
    public FileLock lock(long arg0, long arg1, boolean arg2)
        throws IOException {
        return null;
    }

    /**
     * @see java.nio.channels.FileChannel#map(MapMode, long, long)
     */
    public MappedByteBuffer map(MapMode arg0, long arg1, long arg2)
        throws IOException {
        return null;
    }

    /**
     * @see java.nio.channels.FileChannel#tryLock(long, long, boolean)
     */
    public FileLock tryLock(long arg0, long arg1, boolean arg2)
        throws IOException {
        return null;
    }

}
