// TypeCheckTest.java, created Jul 6, 2003 2:53:03 AM by John Whaley
// Copyright (C) 2003 John Whaley
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package Run_Time;

import junit.framework.TestCase;

/**
 * Tests for Run_Time.TypeCheck 
 * 
 * @author John Whaley
 * @version $Id$
 */
public class TypeCheckTest extends TestCase {

    /**
     * Constructor for TypeCheckTest.
     * @param arg0
     */
    public TypeCheckTest(String arg0) {
        super(arg0);
    }

    public static void main(String[] args) {
        junit.textui.TestRunner.run(TypeCheckTest.class);
        new TypeCheckTest("Time").timeAll();
    }

    java.util.List typeList;
    
    Clazz.jq_Class c_jlo; // java.lang.Object
    Clazz.jq_Class c_jls; // java.lang.String
    Clazz.jq_Class c_jis; // java.io.Serializable
    Clazz.jq_Class c_jlc; // java.lang.Cloneable
    Clazz.jq_Class c_jnn; // javax.naming.Name (subinterface of Cloneable)
    Clazz.jq_Class c_jnc; // javax.naming.CompositeName (implements Name)
    Clazz.jq_Class c_jus; // java.util.Stack (implements Cloneable)
    
    Clazz.jq_Array a_I;   // [I
    Clazz.jq_Array a_F;   // [F
    Clazz.jq_Array a_D;   // [D
    Clazz.jq_Array a_Z;   // [Z
    Clazz.jq_Array a_B;   // [B
    Clazz.jq_Array a_jlo; // [Ljava/lang/Object;
    Clazz.jq_Array a_jls; // [Ljava/lang/String;
    Clazz.jq_Array a_jis; // [Ljava/io/Serializable;
    Clazz.jq_Array a_jlc; // [Ljava/lang/Cloneable;
    Clazz.jq_Array a_jnn; // [Ljavax/naming/Name;
    Clazz.jq_Array a_jnc; // [Ljavax/naming/CompositeName;
    Clazz.jq_Array a_jus; // [Ljava/util/Stack;

    Object o_jlo;
    Object o_jls1, o_jls2;
    Object o_jnc;
    Object o_jus;

    Object ao_I;   // [I
    Object ao_F;   // [F
    Object ao_D;   // [D
    Object ao_Z;   // [Z
    Object ao_B;   // [B
    Object ao_jlo; // [Ljava/lang/Object;
    Object ao_jls; // [Ljava/lang/String;
    Object ao_jis; // [Ljava/io/Serializable;
    Object ao_jlc; // [Ljava/lang/Cloneable;
    Object ao_jnn; // [Ljavax/naming/Name;
    Object ao_jnc; // [Ljavax/naming/CompositeName;
    Object ao_jus; // [Ljava/util/Stack;
    
    /*
     * @see TestCase#setUp()
     */
    protected void setUp() throws Exception {
        super.setUp();
        Main.HostedVM.initialize();
        
        c_jlo = Bootstrap.PrimordialClassLoader.getJavaLangObject();
        c_jls = Bootstrap.PrimordialClassLoader.getJavaLangString();
        c_jis = (Clazz.jq_Class) Bootstrap.PrimordialClassLoader.loader.getOrCreateBSType("Ljava/io/Serializable;");
        c_jlc = (Clazz.jq_Class) Bootstrap.PrimordialClassLoader.loader.getOrCreateBSType("Ljava/lang/Cloneable;");
        c_jnn = (Clazz.jq_Class) Bootstrap.PrimordialClassLoader.loader.getOrCreateBSType("Ljavax/naming/Name;");
        c_jnc = (Clazz.jq_Class) Bootstrap.PrimordialClassLoader.loader.getOrCreateBSType("Ljavax/naming/CompositeName;");
        c_jus = (Clazz.jq_Class) Bootstrap.PrimordialClassLoader.loader.getOrCreateBSType("Ljava/util/Stack;");
        
        a_I = Clazz.jq_Array.INT_ARRAY;
        a_F = Clazz.jq_Array.FLOAT_ARRAY;
        a_D = Clazz.jq_Array.DOUBLE_ARRAY;
        a_Z = Clazz.jq_Array.BOOLEAN_ARRAY;
        a_B = Clazz.jq_Array.BYTE_ARRAY;
        a_jlo = c_jlo.getArrayTypeForElementType();
        a_jls = c_jls.getArrayTypeForElementType();
        a_jis = c_jis.getArrayTypeForElementType();
        a_jlc = c_jlc.getArrayTypeForElementType();
        a_jnn = c_jnn.getArrayTypeForElementType();
        a_jnc = c_jnc.getArrayTypeForElementType();
        a_jus = c_jus.getArrayTypeForElementType();
        
        o_jlo = new Object();
        o_jls1 = new String();
        o_jls2 = "foo";
        o_jnc = new javax.naming.CompositeName();
        o_jus = new java.util.Stack();
        
        ao_I = new int[1];
        ao_F = new float[1];
        ao_D = new double[1];
        ao_Z = new boolean[1];
        ao_B = new byte[1];
        ao_jlo = new Object[1];
        ao_jls = new String[1];
        ao_jis = new java.io.Serializable[1];
        ao_jlc = new Cloneable[1];
        ao_jnn = new javax.naming.Name[1];
        ao_jnc = new javax.naming.CompositeName[1];
        ao_jus = new java.util.Stack[1];
    }

    public void testCheckcast() {
        Run_Time.TypeCheck.checkcast(o_jlo, c_jlo);
        Run_Time.TypeCheck.checkcast(o_jls1, c_jlo);
        Run_Time.TypeCheck.checkcast(o_jls2, c_jlo);
        Run_Time.TypeCheck.checkcast(o_jnc, c_jlo);
        Run_Time.TypeCheck.checkcast(ao_I, c_jlo);
        Run_Time.TypeCheck.checkcast(ao_Z, c_jlo);
        Run_Time.TypeCheck.checkcast(ao_jlo, c_jlo);
        Run_Time.TypeCheck.checkcast(ao_jnn, c_jlo);
        
        Run_Time.TypeCheck.checkcast(o_jls1, c_jls);
        Run_Time.TypeCheck.checkcast(o_jls2, c_jls);
        
        // make sure arrays implement java.lang.Cloneable
        Run_Time.TypeCheck.checkcast(o_jnc, c_jlc);
        Run_Time.TypeCheck.checkcast(ao_I, c_jlc);
        Run_Time.TypeCheck.checkcast(ao_F, c_jlc);
        Run_Time.TypeCheck.checkcast(ao_D, c_jlc);
        Run_Time.TypeCheck.checkcast(ao_Z, c_jlc);
        Run_Time.TypeCheck.checkcast(ao_B, c_jlc);
        Run_Time.TypeCheck.checkcast(ao_jlo, c_jlc);
        Run_Time.TypeCheck.checkcast(ao_jlc, c_jlc);
        Run_Time.TypeCheck.checkcast(ao_jls, c_jlc);
        
        // make sure arrays implement java.io.Serializable
        Run_Time.TypeCheck.checkcast(ao_I, c_jis);
        Run_Time.TypeCheck.checkcast(ao_F, c_jis);
        Run_Time.TypeCheck.checkcast(ao_D, c_jis);
        Run_Time.TypeCheck.checkcast(ao_Z, c_jis);
        Run_Time.TypeCheck.checkcast(ao_B, c_jis);
        Run_Time.TypeCheck.checkcast(ao_jlo, c_jis);
        Run_Time.TypeCheck.checkcast(ao_jlc, c_jis);
        Run_Time.TypeCheck.checkcast(ao_jls, c_jis);
        Run_Time.TypeCheck.checkcast(ao_jis, c_jis);
        
        Run_Time.TypeCheck.checkcast(o_jnc, c_jnn);
        Run_Time.TypeCheck.checkcast(o_jnc, c_jlc);
        
        try {
            Run_Time.TypeCheck.checkcast(ao_I, c_jnn);
            fail();
        } catch (ClassCastException _) { }
        try {
            Run_Time.TypeCheck.checkcast(ao_jnn, c_jnn);
            fail();
        } catch (ClassCastException _) { }
        try {
            Run_Time.TypeCheck.checkcast(o_jls1, c_jnn);
            fail();
        } catch (ClassCastException _) { }
        try {
            Run_Time.TypeCheck.checkcast(o_jls2, c_jnn);
            fail();
        } catch (ClassCastException _) { }
        try {
            Run_Time.TypeCheck.checkcast(o_jls2, c_jnc);
            fail();
        } catch (ClassCastException _) { }
        try {
            Run_Time.TypeCheck.checkcast(o_jlo, c_jlc);
            fail();
        } catch (ClassCastException _) { }
        try {
            Run_Time.TypeCheck.checkcast(o_jlo, c_jis);
            fail();
        } catch (ClassCastException _) { }
    }

    public void testInstance_of() {
        assertTrue(Run_Time.TypeCheck.instance_of(ao_jlo, c_jis));
        assertFalse(Run_Time.TypeCheck.instance_of(null, c_jis));
        assertFalse(Run_Time.TypeCheck.instance_of(null, c_jlc));
        assertFalse(Run_Time.TypeCheck.instance_of(null, c_jlo));
        assertFalse(Run_Time.TypeCheck.instance_of(null, a_I));
        assertFalse(Run_Time.TypeCheck.instance_of(null, a_jlc));
    }

    public void testIsSuperclassOf() {
        assertTrue(Run_Time.TypeCheck.isSuperclassOf(c_jlo, c_jls));
        assertTrue(Run_Time.TypeCheck.isSuperclassOf(c_jlo, c_jnn));
        assertFalse(Run_Time.TypeCheck.isSuperclassOf(c_jls, c_jlo));
        assertFalse(Run_Time.TypeCheck.isSuperclassOf(c_jls, c_jnc));
        assertFalse(Run_Time.TypeCheck.isSuperclassOf(c_jls, c_jnn));
    }

    public void testAllTypes() {
        prepareAll();
        checkAll();
    }
    
    public void prepareAll() {
        typeList = new java.util.LinkedList(Bootstrap.PrimordialClassLoader.loader.getAllTypes());
        java.util.Iterator i = typeList.iterator();
        while (i.hasNext()) {
            Clazz.jq_Type t = (Clazz.jq_Type) i.next();
            t.prepare();
        }
        typeList = new java.util.LinkedList(Bootstrap.PrimordialClassLoader.loader.getAllTypes());
    }
    
    public void checkAll() {
        java.util.Iterator i1 = typeList.iterator();
        while (i1.hasNext()) {
            Clazz.jq_Type t1 = (Clazz.jq_Type) i1.next();
            t1.prepare();
            java.util.Iterator i2 = typeList.iterator();
            while (i2.hasNext()) {
                Clazz.jq_Type t2 = (Clazz.jq_Type) i2.next();
                t2.prepare();
                check(t1, t2);
            }
        }
    }
    
    public static void check(Clazz.jq_Type t1, Clazz.jq_Type t2) {
        if (t1 instanceof Clazz.jq_Reference && t2 instanceof Clazz.jq_Reference) {
            Clazz.jq_Reference r1 = (Clazz.jq_Reference) t1;
            Clazz.jq_Reference r2 = (Clazz.jq_Reference) t2;
            boolean b1 = r1.isSubtypeOf(r2);
            boolean b2 = TypeCheck.isAssignable_graph(r1, r2);
            boolean b3 = r2.getJavaLangClassObject().isAssignableFrom(r1.getJavaLangClassObject());
            String s = r1.toString()+","+r2.toString();
            assertEquals(s, b1, b3);
            assertEquals(s, b2, b3);
        }
    }

    public void timeAll() {
        
        prepareAll();
        System.out.println(typeList.size()+" types");
        
        int total;
        long time;
        double sec;
        
        total = 0;
        time = System.currentTimeMillis();
        for (java.util.Iterator i1 = typeList.iterator(); i1.hasNext(); ) {
            Clazz.jq_Type t1 = (Clazz.jq_Type) i1.next();
            t1.prepare();
            for (java.util.Iterator i2 = typeList.iterator(); i2.hasNext(); ) {
                Clazz.jq_Type t2 = (Clazz.jq_Type) i2.next();
                t2.prepare();
                total++;
                TypeCheck.isAssignable_graph(t1, t2);
            }
        }
        time = System.currentTimeMillis() - time;
        sec = time/1000.;
        System.out.println(total+" unique type checks using graph method: "+sec+" seconds, "+(total/sec)+" per second.");
        
        total = 0;
        time = System.currentTimeMillis();
        for (java.util.Iterator i1 = typeList.iterator(); i1.hasNext(); ) {
            Clazz.jq_Type t1 = (Clazz.jq_Type) i1.next();
            t1.prepare();
            for (java.util.Iterator i2 = typeList.iterator(); i2.hasNext(); ) {
                Clazz.jq_Type t2 = (Clazz.jq_Type) i2.next();
                t2.prepare();
                total++;
                TypeCheck.isAssignable(t1, t2);
            }
        }
        time = System.currentTimeMillis() - time;
        sec = time/1000.;
        System.out.println(total+" unique type checks using fast method: "+sec+" seconds, "+(total/sec)+" per second.");
        
    }
}
