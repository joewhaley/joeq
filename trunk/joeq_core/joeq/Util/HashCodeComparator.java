package Util;

import java.lang.ref.WeakReference;
import java.util.Comparator;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

import Main.jq;

/**
 * @author John Whaley
 */
public class HashCodeComparator implements Comparator {

    public static final boolean USE_IDENTITY_HASHCODE = false;
    public static final boolean USE_WEAK_REFERENCES = true;
    
    public static final HashCodeComparator INSTANCE = new HashCodeComparator();

    public HashCodeComparator() { }

    private final List duplicate_hashcode_objects = new LinkedList();

    /**
     * @see java.util.Comparator#compare(java.lang.Object, java.lang.Object)
     */
    public int compare(Object arg0, Object arg1) {
        boolean eq;
        if (USE_IDENTITY_HASHCODE) eq = arg0 == arg1;
        else eq = arg0.equals(arg1);
        if (eq) return 0;
        int a, b;
        if (USE_IDENTITY_HASHCODE) {
            a = System.identityHashCode(arg0);
            b = System.identityHashCode(arg1);
        } else {
            a = arg0.hashCode();
            b = arg1.hashCode();
        }
        if (a > b) return 1;
        if (a < b) return -1;
        if (USE_IDENTITY_HASHCODE) {
            arg0 = IdentityHashCodeWrapper.create(arg0);
            arg1 = IdentityHashCodeWrapper.create(arg1);
        }
        int i1 = indexOf(arg0);
        if (i1 == -1) {
            i1 = duplicate_hashcode_objects.size();
            if (USE_WEAK_REFERENCES) arg0 = new WeakReference(arg0);
            duplicate_hashcode_objects.add(arg0);
        }
        int i2 = indexOf(arg1);
        if (i1 < i2) return -1;
        if (i2 == -1) {
            i2 = duplicate_hashcode_objects.size();
            if (USE_WEAK_REFERENCES) arg1 = new WeakReference(arg1);
            duplicate_hashcode_objects.add(arg1);
        }
        if (i1 > i2) return 1;
        jq.Assert(i1 != i2);
        return -1;
    }

    private int indexOf(Object o) {
        if (!USE_WEAK_REFERENCES)
            return duplicate_hashcode_objects.indexOf(o);
        int index = 0;
        for (Iterator i=duplicate_hashcode_objects.iterator(); i.hasNext(); ++index) {
            WeakReference r = (WeakReference) i.next();
            if (o.equals(r.get())) return index;
        }
        return -1;
    }

}
