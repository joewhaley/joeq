/**
 * StackHeapWalker
 *
 * Created on Sep 26, 2002, 8:47:02 PM
 *
 * @author laudney <bin_ren@myrealbox.com>
 * @version 0.1
 */
package Run_Time;

import java.util.ArrayList;

import GC.GCBitsManager;
import Memory.HeapAddress;
import Memory.StackAddress;

public class StackHeapWalker {

    public static /*final*/ boolean TRACE = false;

    // Since not everything on a stack is a reference to an object on the heap,
    // efforts have been taken to make sure hp is either a valid HeapAddress or null.
    private HeapAddress hp;
    private StackAddress fp, sp;
    private ArrayList validHeapAddrs = new ArrayList();

    private boolean gotoNext() {
        if (sp.difference(fp) != 0) { // sp should alwasys be equal to or smaller than fp
            sp = (StackAddress) sp.offset(4);
        }

        if (sp.difference(fp) != 0) {
            if (TRACE) SystemInterface.debugwriteln("StackHeapWalker next: fp=" + fp.stringRep() + " sp=" + sp.stringRep());
            return true; // successful
        } else {
            do {
                fp = (StackAddress) fp.peek();
                if (fp.isNull()) return false; // failed
                sp = (StackAddress) sp.offset(8); // skipping return address
            } while (sp.difference(fp) == 0);
            if (TRACE) SystemInterface.debugwriteln("StackHeapWalker next: fp=" + fp.stringRep() + " sp=" + sp.stringRep());
            return true;
        }
    }


    private void scan() {
        if (sp == null || fp == null) {
            return;
        }

        HeapAddress addr;
        do {
            addr = (HeapAddress) sp.peek();
            if (GCBitsManager.isValidHeapAddr(addr)) {
                validHeapAddrs.add(addr);
            }
        } while (gotoNext());
    }

    public StackHeapWalker(StackAddress sp, StackAddress fp) {
        this.sp = sp;
        this.fp = fp;
        hp = null;
        if (sp == null || fp == null || sp.isNull() || fp.isNull() || sp.difference(fp) > 0) { // invalid passing arguments
            this.sp = this.fp = null;
        } else if (sp.difference(fp) == 0) { // to ensure from the very beginning that sp < fp
            if (!gotoNext()) {
                sp = fp = null;
            }
        }
        // if reach this point, sp and fp are:
        // (1) both null if no valid values can be found for them
        // (2) both valid, i.e. either isNull() returns false and sp < fp
        // thus, sp.peek4() should return a qualified candidate for HeapAddress verification
        scan();

        if (TRACE) SystemInterface.debugwriteln("StackHeapWalker init: fp=" + fp.stringRep() + " sp=" + sp.stringRep());
    }

    // totally 3 conditions must be met
    public boolean isValidHeapAddr(HeapAddress addr) {
        return GCBitsManager.isValidHeapAddr(addr);
    }

    public ArrayList getValidHeapAddrs() {
        return validHeapAddrs;
    }

}
