package Util.Collections;

/**
 * @author John Whaley
 * @version $Id$
 */
public class Pair extends java.util.AbstractList
    implements java.io.Serializable {
    public Object left, right;
    public Pair(Object left, Object right) {
        this.left = left; this.right = right;
    }
    public int size() { return 2; }
    public Object get(int index) {
        switch(index) {
        case 0: return this.left;
        case 1: return this.right;
        default: throw new IndexOutOfBoundsException();
        }
    }
    public Object set(int index, Object element) {
        Object prev;
        switch(index) {
        case 0: prev=this.left; this.left=element; return prev;
        case 1: prev=this.right; this.right=element; return prev;
        default: throw new IndexOutOfBoundsException();
        }
    }
}
