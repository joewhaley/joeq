/*
 * Templates.java
 *
 * Created on January 15, 2002, 5:15 PM
 */

package Util;

/**
 *
 * @author  John Whaley
 * @version 
 */
public abstract class Templates {

    public static abstract class List {
        public static interface jq_Type extends java.util.List {
            public Clazz.jq_Type getType(int index);
            public ListIterator.jq_Type typeIterator();
        }
        public static interface jq_Reference extends jq_Type {
            public Clazz.jq_Reference getReference(int index);
            public ListIterator.jq_Reference referenceIterator();
        }
        public static interface jq_Class extends jq_Reference {
            public Clazz.jq_Class getClass(int index);
            public ListIterator.jq_Class classIterator();
        }
        public static interface jq_Member extends java.util.List {
            public Clazz.jq_Member getMember(int index);
            public ListIterator.jq_Member memberIterator();
        }
        public static interface jq_Method extends jq_Member {
            public Clazz.jq_Method getMethod(int index);
            public ListIterator.jq_Method methodIterator();
        }
        public static interface jq_InstanceMethod extends jq_Method {
            public Clazz.jq_InstanceMethod getInstanceMethod(int index);
            public ListIterator.jq_InstanceMethod instanceMethodIterator();
        }
        public static interface jq_StaticMethod extends jq_Method {
            public Clazz.jq_StaticMethod getStaticMethod(int index);
            public ListIterator.jq_StaticMethod staticMethodIterator();
        }
    }
    
    public static abstract class ListIterator {
        public static interface jq_Type extends java.util.ListIterator {
            public Clazz.jq_Type nextType();
            public Clazz.jq_Type previousType();
        }
        public static interface jq_Reference extends jq_Type {
            public Clazz.jq_Reference nextReference();
            public Clazz.jq_Reference previousReference();
        }
        public static interface jq_Class extends jq_Reference {
            public Clazz.jq_Class nextClass();
            public Clazz.jq_Class previousClass();
        }
        public static interface jq_Member extends java.util.ListIterator {
            public Clazz.jq_Member nextMember();
            public Clazz.jq_Member previousMember();
        }
        public static interface jq_Method extends jq_Member {
            public Clazz.jq_Method nextMethod();
            public Clazz.jq_Method previousMethod();
        }
        public static interface jq_InstanceMethod extends jq_Method {
            public Clazz.jq_InstanceMethod nextInstanceMethod();
            public Clazz.jq_InstanceMethod previousInstanceMethod();
        }
        public static interface jq_StaticMethod extends jq_Method {
            public Clazz.jq_StaticMethod nextStaticMethod();
            public Clazz.jq_StaticMethod previousStaticMethod();
        }
    }
    
    public static abstract class UnmodifiableList {
        public static class jq_Class extends java.util.AbstractList implements List.jq_Class {
            private final Clazz.jq_Class[] a;
            public jq_Class(Clazz.jq_Class c) { a = new Clazz.jq_Class[] { c }; }
            public jq_Class(Clazz.jq_Class c1, Clazz.jq_Class c2) { a = new Clazz.jq_Class[] { c1, c2 }; }
            public jq_Class(Clazz.jq_Class c1, Clazz.jq_Class c2, Clazz.jq_Class c3) { a = new Clazz.jq_Class[] { c1, c2, c3 }; }
            public jq_Class(Clazz.jq_Class[] c) { a = c; }
            public int size() { return a.length; }
            public Object get(int index) { return getClass(index); }
            public Clazz.jq_Type getType(int index) { return a[index]; }
            public Clazz.jq_Reference getReference(int index) { return a[index]; }
            public Clazz.jq_Class getClass(int index) { return a[index]; }
            public ListIterator.jq_Type typeIterator() { return new Iterator(); }
            public ListIterator.jq_Reference referenceIterator() { return new Iterator(); }
            public ListIterator.jq_Class classIterator() { return new Iterator(); }
            private class Iterator extends UnmodifiableListIterator implements ListIterator.jq_Class {
                private int i = -1;
                public boolean hasNext() { return i+1 < a.length; }
                public boolean hasPrevious() { return i >= 0; }
                public int nextIndex() { return i+1; }
                public int previousIndex() { return i; }
                public Object next() { return a[++i]; }
                public Clazz.jq_Type nextType() { return a[++i]; }
                public Clazz.jq_Reference nextReference() { return a[++i]; }
                public Clazz.jq_Class nextClass() { return a[++i]; }
                public Object previous() { return a[i--]; }
                public Clazz.jq_Type previousType() { return a[i--]; }
                public Clazz.jq_Reference previousReference() { return a[i--]; }
                public Clazz.jq_Class previousClass() { return a[i--]; }
            }
        }
    }
}
