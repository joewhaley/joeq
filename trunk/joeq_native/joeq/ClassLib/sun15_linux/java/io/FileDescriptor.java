// FileDescriptor.java, created Fri Apr  5 18:36:41 2002 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.ClassLib.sun15_linux.java.io;

/*
 * @author  John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
class FileDescriptor {
    int fd;
    private long handle;
    
    static FileDescriptor in;
    static FileDescriptor out;
    static FileDescriptor err;
    
    private /* */ FileDescriptor(int fd) {
	this.fd = fd;
    }
    
    public static void init() {
        in = new FileDescriptor(0);
        out = new FileDescriptor(0);
        err = new FileDescriptor(0);
        //in.fd = 0;
        //out.fd = 1;
        //err.fd = 2;
    }
    
}
