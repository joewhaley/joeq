// DebugImpl.java, created Sat Feb 22 13:35:27 2003 by joewhaley
// Copyright (C) 2001-3 mcmartin
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package Run_Time;

/**
 * @author  Michael Martin <mcmartin@stanford.edu>
 * @version $Id$
 */
public class DebugImpl implements Debug.Delegate {

    public void write(byte[] msg, int size) {
        SystemInterface.debugwrite(msg, size);
    }

    public void write(String msg) {
        SystemInterface.debugwrite(msg);
    }

    public void writeln(byte[] msg, int size) {
        SystemInterface.debugwriteln(msg, size);
    }

    public void writeln(String msg) {
        SystemInterface.debugwriteln(msg);
    }

    public void die(int code) {
        if (code != 0)
            Debugger.OnlineDebugger.debuggerEntryPoint();
        SystemInterface.die(code);
    }

}
