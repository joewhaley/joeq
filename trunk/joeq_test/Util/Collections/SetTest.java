// SetTest.java, created Jul 6, 2003 2:53:03 AM by John Whaley
// Copyright (C) 2003 John Whaley
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package Util.Collections;

import junit.framework.TestCase;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import java.util.Vector;

/**
 * The <code>SetTest</code> class tests a <code>Set</code>
 * implementation for correctness.  Large portions borrowed from Mauve.
 * 
 * @author  John Whaley
 * @version $Id$
 */
public class SetTest extends TestCase {

    /**
     * Constructor for SetTest.
     * @param arg0
     */
    public SetTest(String arg0) {
        super(arg0);
    }

    public static void main(String[] args) {
        junit.textui.TestRunner.run(SetTest.class);
    }

    Object[] universe =
        new Object[] {
            null,
            "a",
            "b",
            "c",
            "d",
            "smartmove",
            "rules",
            "cars",
            self() };
            
    // override this method to test other set representations.
    public Set makeSet() {
        NULL = "Not really null";
        return SortedArraySet.FACTORY.makeSet();
    }

    Object self() { return this; }
    
    // this next field can be changed for impl's w/ problems w/ real 'null'
    public static String NULL = null;

    // from AcuniaAbstractSetTest.java: ///////////////////////////////

    /**
    * implemented. <br>
    *
    */
    public void test_equals() {
        Set xas1 = makeSet();
        Set xas2 = makeSet();
        assertTrue(xas1.equals(xas2));
        assertTrue(!xas1.equals(null));
        assertTrue(!xas1.equals(this));
        assertTrue(xas1.equals(xas1));
        xas1.add(NULL);
        xas1.add("a");
        xas2.add("b");
        xas2.add(NULL);
        xas2.add("a");
        xas1.add("b");
        assertTrue(xas1.equals(xas2));
        assertTrue(xas1.equals(xas1));

    }
    /**
    * implemented. <br>
    *
    */
    public void test_hashCode() {
        Set xas = makeSet();
        assertEquals(xas.hashCode(), 0);
        if (NULL == null) {
            xas.add(NULL);
            assertEquals(xas.hashCode(), 0);
        }
        xas.add("a");
        int hash = "a".hashCode();
        assertEquals(xas.hashCode(), hash);
        hash += "b".hashCode();
        xas.add("b");
        assertEquals(xas.hashCode(), hash);
        hash += "c".hashCode();
        xas.add("c");
        assertEquals(xas.hashCode(), hash);
        hash += "d".hashCode();
        xas.add("d");
        assertEquals(xas.hashCode(), hash);
    }

    // from AcuniaAbstractCollectionTest.java: ///////////////////////////////
    /**
    *  implemented. <br>
    *
    */
    public void test_add() {
        Set ac = makeSet();
        if (!ac.add(self()))
            fail("should return true.");
    }

    /**
    * implemented. <br>
    *
    */
    public void test_addAll() {
        Vector v = new Vector();
        v.add("a");
        v.add("b");
        v.add("c");
        v.add("d");
        Set ac = makeSet();
        assertTrue(ac.addAll(v));
        assertEquals(ac, new HashSet(v));
        try {
            ac.addAll(null);
            fail("should throw a NullPointerException");
        } catch (NullPointerException ne) {
            assertTrue(true);
        }
    }

    /**
    * implemented. <br>
    *
    */
    public void test_clear() {
        Set ac = makeSet();
        ac.add("a");
        ac.add("b");
        ac.add("c");
        ac.add("d");
        ac.clear();
        assertEquals(ac.size(), 0);
        ac.clear();
        assertEquals(ac.size(), 0);
    }

    /**
    * implemented. <br>
    *
    */
    public void test_remove() {
        Set ac = makeSet();
        ac.add("a");
        ac.add(NULL);
        ac.add("c");
        ac.add("a");
        assertEquals(ac.size(), 3);
        assertTrue(ac.remove("a"));
        assertEquals(ac.size(), 2);
        assertTrue(!ac.remove("a"));
        assertEquals(ac.size(), 2);
        assertTrue(ac.remove(NULL));
        assertEquals(ac.size(), 1);
        assertTrue(!ac.remove(NULL));
        assertEquals(ac.size(), 1);
        assertTrue(ac.contains("c"));
        assertTrue(ac.remove("c"));
        assertEquals(ac.size(), 0);
    }

    /**
    * implemented. <br>
    *
    */
    public void test_removeAll() {
        Set ac = makeSet();
        ac.add("a");
        ac.add(NULL);
        ac.add("c");
        ac.add("a");
        try {
            ac.removeAll(null);
            fail("should throw a NullPointerException");
        } catch (NullPointerException ne) {
            assertTrue(true);
        }
        Vector v = new Vector();
        v.add("a");
        v.add(NULL);
        v.add("de");
        v.add("fdf");
        assertTrue(ac.removeAll(v));
        assertEquals(ac.size(), 1);
        assertTrue(ac.contains("c"));
        assertTrue(!ac.removeAll(v));
        assertEquals(ac.size(), 1);

    }

    /**
    * implemented. <br>
    *
    */
    public void test_retainAll() {
        Set ac = makeSet();
        ac.add("a");
        ac.add(NULL);
        ac.add("c");
        ac.add("a");
        assertEquals(ac.size(), 3);
        try {
            ac.retainAll(null);
            fail("should throw a NullPointerException");
        } catch (NullPointerException ne) {
            assertTrue(true);
        }
        Vector v = new Vector();
        v.add("a");
        v.add(NULL);
        v.add("de");
        v.add("fdf");
        assertTrue(ac.retainAll(v));
        assertEquals(ac.size(), 2);
        assertTrue(!ac.retainAll(v));
        assertEquals(ac.size(), 2);
        assertTrue(ac.contains(NULL) && ac.contains("a"));
    }

    /**
    * implemented. <br>
    *
    */
    public void test_contains() {
        Set ac = makeSet();
        ac.add("a");
        ac.add(NULL);
        ac.add("c");
        ac.add("a");
        assertTrue(ac.contains("a"));
        assertTrue(ac.contains(NULL));
        assertTrue(ac.contains("c"));
        assertTrue(!ac.contains("ab"));
        assertTrue(!ac.contains("b"));
        ac.remove(NULL);
        assertTrue(!ac.contains(NULL));

    }

    /**
    * implemented. <br>
    *
    */
    public void test_containsAll() {
        Set ac = makeSet();
        ac.add("a");
        ac.add(NULL);
        ac.add("c");
        ac.add("a");
        try {
            ac.containsAll(null);
            fail("should throw a NullPointerException");
        } catch (NullPointerException ne) {
            assertTrue(true);
        }
        Vector v = new Vector();
        assertTrue(ac.containsAll(v));
        v.add("a");
        v.add(NULL);
        v.add("a");
        v.add(NULL);
        v.add("a");
        assertTrue(ac.containsAll(v));
        v.add("c");
        assertTrue(ac.containsAll(v));
        v.add("c+");
        assertTrue(!ac.containsAll(v));
        v.clear();
        ac.clear();
        assertTrue(ac.containsAll(v));

    }

    /**
    * implemented. <br>
    *
    */
    public void test_isEmpty() {
        Set ac = makeSet();
        assertTrue(ac.isEmpty());
        assertTrue(ac.isEmpty());
        ac.add(NULL);
        assertTrue(!ac.isEmpty());
        ac.clear();
        assertTrue(ac.isEmpty());
    }

    /**
    *   not implemented. <br>
    *   Abstract Method
    */
    public void test_size() {
    }
    /**
    *   not implemented. <br>
    *   Abstract Method
    */
    public void test_iterator() {
    }

    /**
    * implemented. <br>
    *
    */
    public void test_toArray() {
        Set ac = makeSet();
        Object[] oa = ac.toArray();
        assertTrue(oa != null);
        if (oa != null)
        assertEquals(oa.length, 0);
        ac.add("a");
        ac.add(NULL);
        ac.add("c");
        ac.add("a");
        assertEquals(ac.size(), 3);
        oa = ac.toArray();
        assertEquals(oa.length, 3);
        assertTrue(Arrays.asList(oa).contains("a"));
        assertTrue(Arrays.asList(oa).contains("c"));
        assertTrue(Arrays.asList(oa).contains(NULL));

        try {
            ac.toArray(null);
            fail("should throw a NullPointerException");
        } catch (NullPointerException ne) {
            assertTrue(true);
        }
        String[] sa = new String[4];
        for (int i = 0; i < sa.length; i++) {
            sa[i] = "ok";
        }
        oa = ac.toArray(sa);
        assertTrue(oa.length >= 3);
        assertTrue(Arrays.asList(oa).contains("a"));
        assertTrue(Arrays.asList(oa).contains("c"));
        assertTrue(Arrays.asList(oa).contains(NULL));
        assertTrue(!Arrays.asList(oa).contains("ok"));
        assertTrue(oa == sa);
        assertEquals(sa[3], null);

        sa = new String[2];
        for (int i = 0; i < sa.length; i++) {
            sa[i] = "ok";
        }
        oa = ac.toArray(sa);
        assertTrue(oa.length >= 3);
        assertTrue(Arrays.asList(oa).contains("a"));
        assertTrue(Arrays.asList(oa).contains("c"));
        assertTrue(Arrays.asList(oa).contains(NULL));
        assertTrue(oa instanceof String[]);
        sa = new String[3];
        Class asc = sa.getClass();
        for (int i = 0; i < sa.length; i++) {
            sa[i] = "ok";
        }
        oa = ac.toArray(sa);
        assertTrue(oa.length >= 3);
        assertTrue(Arrays.asList(oa).contains("a"));
        assertTrue(Arrays.asList(oa).contains("c"));
        assertTrue(Arrays.asList(oa).contains(NULL));
        assertTrue(oa instanceof String[]);
        assertTrue(oa == sa);
    }
    /**
    * implemented. <br>
    *
    */
    public void test_toString() {
        Set ac = makeSet();
        ac.add("smartmove");
        ac.add(NULL);
        ac.add("rules");
        ac.add("cars");
        String s = ac.toString();
        assertTrue(s, s.indexOf("smartmove") != -1);
        assertTrue(s, s.indexOf("rules") != -1);
        assertTrue(s, s.indexOf("cars") != -1);
        assertTrue(s, s.indexOf("null") != -1);
    }
}
