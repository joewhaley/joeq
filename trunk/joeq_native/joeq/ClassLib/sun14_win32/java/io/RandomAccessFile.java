package ClassLib.sun14_win32.java.io;

import Run_Time.SystemInterface;

public class RandomAccessFile {

    private FileDescriptor fd;
    
    private static final int O_RDONLY = 1;
    private static final int O_RDWR =   2;
    private static final int O_SYNC =   4;
    private static final int O_DSYNC =  8;

    public void open(java.lang.String name, int mode)
    throws java.io.FileNotFoundException {
        byte[] filename = SystemInterface.toCString(name);
	int flags = SystemInterface._O_BINARY;
	if ((mode & O_RDONLY) != 0) flags |= SystemInterface._O_RDONLY;
	if ((mode & O_RDWR) != 0) flags |= SystemInterface._O_RDWR | SystemInterface._O_CREAT;
	if ((mode & O_SYNC) != 0) {
	    // todo: require that every update to the file's content or metadata be
	    //       written synchronously to the underlying storage device
	}
	if ((mode & O_DSYNC) != 0) {
	    // todo: require that every update to the file's content be written
	    //       synchronously to the underlying storage device.
	}
        int fdnum = SystemInterface.file_open(filename, flags, 0);
        if (fdnum == -1) throw new java.io.FileNotFoundException(name);
        this.fd.fd = fdnum;
    }

}
