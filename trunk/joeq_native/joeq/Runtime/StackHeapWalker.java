/**
 * StackHeapWalker
 *
 * Created on Sep 26, 2002, 8:47:02 PM
 *
 * @author laudney <bin_ren@myrealbox.com>
 * @version 0.1
 */
package Run_Time;

import Memory.HeapAddress;
import Memory.StackAddress;
import Memory.CodeAddress;

import java.util.Iterator;
import java.util.NoSuchElementException;

public class StackHeapWalker implements Iterator {

    public static /*final*/ boolean TRACE = false;

    // Since not everything on a stack is a reference to an object on the heap,
    // efforts have been taken to make sure hp is either a valid HeapAddress or null.
    HeapAddress hp;
    StackAddress fp, sp;

    public HeapAddress getHP() {
        return hp;
    }

    public StackAddress getFP() {
        return fp;
    }

    public StackAddress getSP() {
        return sp;
    }

    public StackHeapWalker(StackAddress sp) {
        this.sp = sp;
        fp = StackAddress.getBasePointer();
        hp = null;
        if (TRACE) SystemInterface.debugmsg("StackHeapWalker init: fp=" + fp.stringRep() + " sp=" + sp.stringRep());
    }

    //PRE-REQUISITE: method equals() is overridden in class StackAddress
    public void gotoNext() throws NoSuchElementException {
        if (!sp.equals(fp)) {
            sp = (StackAddress)sp.offset(4);
        }

        if (!sp.equals(fp)) {
            if (TRACE) SystemInterface.debugmsg("StackHeapWalker next: fp=" + fp.stringRep() + " sp=" + sp.stringRep());
            return;
        } else {
            do {
                fp = (StackAddress)fp.peek();
                if (fp.isNull()) throw new NoSuchElementException();
                sp = (StackAddress)sp.offset(8); // skipping return address
            } while (sp.equals(fp));
            if (TRACE) SystemInterface.debugmsg("StackHeapWalker next: fp=" + fp.stringRep() + " sp=" + sp.stringRep());
            return;
        }
    }

    public boolean hasNext() {
        StackAddress spTemp = sp, fpTemp;

        if (!sp.equals(fp)) {
            spTemp = (StackAddress)sp.offset(4);
        }

       if (!spTemp.equals(fp)) {
           return true;
       } else {
            do {
                fpTemp = (StackAddress)fp.peek();
                if (fpTemp.isNull()) return false;
                spTemp = (StackAddress)spTemp.offset(8); // skipping return address
            } while (spTemp.equals(fpTemp));
            return true;
       }
    }

    public Object next() throws NoSuchElementException {
        gotoNext();
        return hp = (HeapAddress)sp.peek();
    }

    public void remove() throws UnsupportedOperationException {
        throw new UnsupportedOperationException();
    }
}
