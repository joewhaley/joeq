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
        
        public static interface BasicBlock extends java.util.List {
            public Compil3r.Quad.BasicBlock getBasicBlock(int index);
            public ListIterator.BasicBlock basicBlockIterator();
        }
        public static interface ExceptionHandler extends java.util.List {
            public Compil3r.Quad.ExceptionHandler getExceptionHandler(int index);
            public ListIterator.ExceptionHandler exceptionHandlerIterator();
        }
        public static interface Quad extends java.util.List {
            public Compil3r.Quad.Quad getQuad(int index);
            public ListIterator.Quad quadIterator();
        }
        public static interface RegisterOperand extends java.util.List {
            public Compil3r.Quad.Operand.RegisterOperand getRegisterOperand(int index);
            public ListIterator.RegisterOperand registerOperandIterator();
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
        
        public static interface BasicBlock extends java.util.ListIterator {
            public Compil3r.Quad.BasicBlock nextBasicBlock();
            public Compil3r.Quad.BasicBlock previousBasicBlock();
        }
        public static interface ExceptionHandler extends java.util.ListIterator {
            public Compil3r.Quad.ExceptionHandler nextExceptionHandler();
            public Compil3r.Quad.ExceptionHandler previousExceptionHandler();
        }
        public static interface Quad extends java.util.ListIterator {
            public Compil3r.Quad.Quad nextQuad();
            public Compil3r.Quad.Quad previousQuad();
        }
        public static interface RegisterOperand extends java.util.ListIterator {
            public Compil3r.Quad.Operand.RegisterOperand nextRegisterOperand();
            public Compil3r.Quad.Operand.RegisterOperand previousRegisterOperand();
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
            public static final jq_Class EMPTY = new jq_Class(new Clazz.jq_Class[0]);
            public static jq_Class getEmptyList() { return EMPTY; }
        }
        
        public static class RegisterOperand extends java.util.AbstractList implements List.RegisterOperand {
            private final Compil3r.Quad.Operand.RegisterOperand[] a;
            public RegisterOperand(Compil3r.Quad.Operand.RegisterOperand c) { a = new Compil3r.Quad.Operand.RegisterOperand[] { c }; }
            public RegisterOperand(Compil3r.Quad.Operand.RegisterOperand c1, Compil3r.Quad.Operand.RegisterOperand c2) { a = new Compil3r.Quad.Operand.RegisterOperand[] { c1, c2 }; }
            public RegisterOperand(Compil3r.Quad.Operand.RegisterOperand c1, Compil3r.Quad.Operand.RegisterOperand c2, Compil3r.Quad.Operand.RegisterOperand c3) { a = new Compil3r.Quad.Operand.RegisterOperand[] { c1, c2, c3 }; }
            public RegisterOperand(Compil3r.Quad.Operand.RegisterOperand c1, Compil3r.Quad.Operand.RegisterOperand c2, Compil3r.Quad.Operand.RegisterOperand c3, Compil3r.Quad.Operand.RegisterOperand c4) { a = new Compil3r.Quad.Operand.RegisterOperand[] { c1, c2, c3, c4 }; }
            public RegisterOperand(Compil3r.Quad.Operand.RegisterOperand[] c) { a = c; }
            public int size() { return a.length; }
            public Object get(int index) { return getRegisterOperand(index); }
            public Compil3r.Quad.Operand.RegisterOperand getRegisterOperand(int index) { return a[index]; }
            public ListIterator.RegisterOperand registerOperandIterator() { return new Iterator(); }
            private class Iterator extends UnmodifiableListIterator implements ListIterator.RegisterOperand {
                private int i = -1;
                public boolean hasNext() { return i+1 < a.length; }
                public boolean hasPrevious() { return i >= 0; }
                public int nextIndex() { return i+1; }
                public int previousIndex() { return i; }
                public Object next() { return a[++i]; }
                public Compil3r.Quad.Operand.RegisterOperand nextRegisterOperand() { return a[++i]; }
                public Object previous() { return a[i--]; }
                public Compil3r.Quad.Operand.RegisterOperand previousRegisterOperand() { return a[i--]; }
            }
            public static final RegisterOperand EMPTY = new RegisterOperand(new Compil3r.Quad.Operand.RegisterOperand[0]);
            public static RegisterOperand getEmptyList() { return EMPTY; }
        }
        
        public static class BasicBlock extends java.util.AbstractList implements List.BasicBlock {
            private final Compil3r.Quad.BasicBlock[] a;
            public BasicBlock(Compil3r.Quad.BasicBlock[] c) { a = c; }
            public int size() { return a.length; }
            public Object get(int index) { return getBasicBlock(index); }
            public Compil3r.Quad.BasicBlock getBasicBlock(int index) { return a[index]; }
            public ListIterator.BasicBlock basicBlockIterator() { return new Iterator(); }
            private class Iterator extends UnmodifiableListIterator implements ListIterator.BasicBlock {
                private int i = -1;
                public boolean hasNext() { return i+1 < a.length; }
                public boolean hasPrevious() { return i >= 0; }
                public int nextIndex() { return i+1; }
                public int previousIndex() { return i; }
                public Object next() { return a[++i]; }
                public Compil3r.Quad.BasicBlock nextBasicBlock() { return a[++i]; }
                public Object previous() { return a[i--]; }
                public Compil3r.Quad.BasicBlock previousBasicBlock() { return a[i--]; }
            }
            public static final BasicBlock EMPTY = new BasicBlock(new Compil3r.Quad.BasicBlock[0]);
            public static BasicBlock getEmptyList() { return EMPTY; }
        }
        
        public static class Quad extends java.util.AbstractList implements List.Quad {
            private final Compil3r.Quad.Quad[] a;
            public Quad(Compil3r.Quad.Quad[] c) { a = c; }
            public Quad(Compil3r.Quad.Quad c) { a = new Compil3r.Quad.Quad[] { c }; }
            public int size() { return a.length; }
            public Object get(int index) { return getQuad(index); }
            public Compil3r.Quad.Quad getQuad(int index) { return a[index]; }
            public ListIterator.Quad quadIterator() { return new Iterator(); }
            private class Iterator extends UnmodifiableListIterator implements ListIterator.Quad {
                private int i = -1;
                public boolean hasNext() { return i+1 < a.length; }
                public boolean hasPrevious() { return i >= 0; }
                public int nextIndex() { return i+1; }
                public int previousIndex() { return i; }
                public Object next() { return a[++i]; }
                public Compil3r.Quad.Quad nextQuad() { return a[++i]; }
                public Object previous() { return a[i--]; }
                public Compil3r.Quad.Quad previousQuad() { return a[i--]; }
            }
            public static final Quad EMPTY = new Quad(new Compil3r.Quad.Quad[0]);
            public static Quad getEmptyList() { return EMPTY; }
        }
    }

    public static abstract class ListWrapper {
        
        public static class BasicBlock extends java.util.AbstractList implements List.BasicBlock {
            private final java.util.List/*<Compil3r.Quad.BasicBlock>*/ a;
            public BasicBlock(java.util.List/*<Compil3r.Quad.BasicBlock>*/ c) { this.a = c; }
            public int size() { return a.size(); }
            public Object get(int index) { return a.get(index); }
            public Compil3r.Quad.BasicBlock getBasicBlock(int index) { return (Compil3r.Quad.BasicBlock)a.get(index); }
            public void add(int i, Object o) { a.add(i, o); }
            public Object set(int i, Object o) { return a.set(i, o); }
            public Object remove(int i) { return a.remove(i); }
            public ListIterator.BasicBlock basicBlockIterator() { return new Iterator(a.listIterator()); }
            public static class Iterator implements ListIterator.BasicBlock {
                private java.util.ListIterator/*<Compil3r.Quad.BasicBlock>*/ i;
                public Iterator(java.util.ListIterator/*<Compil3r.Quad.BasicBlock>*/ l) { this.i = l; }
                public boolean hasNext() { return i.hasNext(); }
                public boolean hasPrevious() { return i.hasPrevious(); }
                public int nextIndex() { return i.nextIndex(); }
                public int previousIndex() { return i.previousIndex(); }
                public Object next() { return i.next(); }
                public Compil3r.Quad.BasicBlock nextBasicBlock() { return (Compil3r.Quad.BasicBlock)i.next(); }
                public Object previous() { return i.previous(); }
                public Compil3r.Quad.BasicBlock previousBasicBlock() { return (Compil3r.Quad.BasicBlock)i.previous(); }
                public void remove() { i.remove(); }
                public void set(Object o) { i.set(o); }
                public void add(Object o) { i.add(o); }
            }
            public static class EmptyIterator implements ListIterator.BasicBlock {
                private EmptyIterator() {}
                public boolean hasNext() { return false; }
                public boolean hasPrevious() { return false; }
                public int nextIndex() { return 0; }
                public int previousIndex() { return -1; }
                public Object next() { throw new java.util.NoSuchElementException(); }
                public Compil3r.Quad.BasicBlock nextBasicBlock() { throw new java.util.NoSuchElementException(); }
                public Object previous() { throw new java.util.NoSuchElementException(); }
                public Compil3r.Quad.BasicBlock previousBasicBlock() { throw new java.util.NoSuchElementException(); }
                public void remove() { throw new java.lang.IllegalStateException(); }
                public void set(Object o) { throw new java.lang.IllegalStateException(); }
                public void add(Object o) { throw new java.lang.UnsupportedOperationException(); }
                public static EmptyIterator INSTANCE = new EmptyIterator();
            }
        }
        
        public static class Quad extends java.util.AbstractList implements List.Quad {
            private final java.util.List/*<Compil3r.Quad.Quad>*/ a;
            public Quad(java.util.List/*<Compil3r.Quad.Quad>*/ c) { this.a = c; }
            public int size() { return a.size(); }
            public Object get(int index) { return a.get(index); }
            public Compil3r.Quad.Quad getQuad(int index) { return (Compil3r.Quad.Quad)a.get(index); }
            public void add(int i, Object o) { a.add(i, o); }
            public Object set(int i, Object o) { return a.set(i, o); }
            public Object remove(int i) { return a.remove(i); }
            public ListIterator.Quad quadIterator() { return new Iterator(a.listIterator()); }
            public static class Iterator implements ListIterator.Quad {
                private java.util.ListIterator/*<Compil3r.Quad.Quad>*/ i;
                public Iterator(java.util.ListIterator/*<Compil3r.Quad.Quad>*/ l) { this.i = l; }
                public boolean hasNext() { return i.hasNext(); }
                public boolean hasPrevious() { return i.hasPrevious(); }
                public int nextIndex() { return i.nextIndex(); }
                public int previousIndex() { return i.previousIndex(); }
                public Object next() { return i.next(); }
                public Compil3r.Quad.Quad nextQuad() { return (Compil3r.Quad.Quad)i.next(); }
                public Object previous() { return i.previous(); }
                public Compil3r.Quad.Quad previousQuad() { return (Compil3r.Quad.Quad)i.previous(); }
                public void remove() { i.remove(); }
                public void set(Object o) { i.set(o); }
                public void add(Object o) { i.add(o); }
            }
            public static class EmptyIterator implements ListIterator.Quad {
                private EmptyIterator() {}
                public boolean hasNext() { return false; }
                public boolean hasPrevious() { return false; }
                public int nextIndex() { return 0; }
                public int previousIndex() { return -1; }
                public Object next() { throw new java.util.NoSuchElementException(); }
                public Compil3r.Quad.Quad nextQuad() { throw new java.util.NoSuchElementException(); }
                public Object previous() { throw new java.util.NoSuchElementException(); }
                public Compil3r.Quad.Quad previousQuad() { throw new java.util.NoSuchElementException(); }
                public void remove() { throw new java.lang.IllegalStateException(); }
                public void set(Object o) { throw new java.lang.IllegalStateException(); }
                public void add(Object o) { throw new java.lang.UnsupportedOperationException(); }
                public static EmptyIterator INSTANCE = new EmptyIterator();
            }
        }
        
        public static class ExceptionHandler extends java.util.AbstractList implements List.ExceptionHandler {
            private final java.util.List/*<Compil3r.Quad.ExceptionHandler>*/ a;
            public ExceptionHandler(java.util.List/*<Compil3r.Quad.ExceptionHandler>*/ c) { this.a = c; }
            public int size() { return a.size(); }
            public Object get(int index) { return a.get(index); }
            public Compil3r.Quad.ExceptionHandler getExceptionHandler(int index) { return (Compil3r.Quad.ExceptionHandler)a.get(index); }
            public void add(int i, Object o) { a.add(i, o); }
            public Object set(int i, Object o) { return a.set(i, o); }
            public Object remove(int i) { return a.remove(i); }
            public ListIterator.ExceptionHandler exceptionHandlerIterator() { return new Iterator(a.listIterator()); }
            public static class Iterator implements ListIterator.ExceptionHandler {
                private java.util.ListIterator/*<Compil3r.Quad.ExceptionHandler>*/ i;
                public Iterator(java.util.ListIterator/*<Compil3r.Quad.ExceptionHandler>*/ l) { this.i = l; }
                public boolean hasNext() { return i.hasNext(); }
                public boolean hasPrevious() { return i.hasPrevious(); }
                public int nextIndex() { return i.nextIndex(); }
                public int previousIndex() { return i.previousIndex(); }
                public Object next() { return i.next(); }
                public Compil3r.Quad.ExceptionHandler nextExceptionHandler() { return (Compil3r.Quad.ExceptionHandler)i.next(); }
                public Object previous() { return i.previous(); }
                public Compil3r.Quad.ExceptionHandler previousExceptionHandler() { return (Compil3r.Quad.ExceptionHandler)i.previous(); }
                public void remove() { i.remove(); }
                public void set(Object o) { i.set(o); }
                public void add(Object o) { i.add(o); }
            }
            public static class EmptyIterator implements ListIterator.ExceptionHandler {
                private EmptyIterator() {}
                public boolean hasNext() { return false; }
                public boolean hasPrevious() { return false; }
                public int nextIndex() { return 0; }
                public int previousIndex() { return -1; }
                public Object next() { throw new java.util.NoSuchElementException(); }
                public Compil3r.Quad.ExceptionHandler nextExceptionHandler() { throw new java.util.NoSuchElementException(); }
                public Object previous() { throw new java.util.NoSuchElementException(); }
                public Compil3r.Quad.ExceptionHandler previousExceptionHandler() { throw new java.util.NoSuchElementException(); }
                public void remove() { throw new java.lang.IllegalStateException(); }
                public void set(Object o) { throw new java.lang.IllegalStateException(); }
                public void add(Object o) { throw new java.lang.UnsupportedOperationException(); }
                public static EmptyIterator INSTANCE = new EmptyIterator();
            }
        }
    }
}
