// ListWrapper.java, created Wed Mar  5  0:26:32 2003 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Util.Templates;

/**
 * @author John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public abstract class ListWrapper {
        
    public static class BasicBlock extends java.util.AbstractList implements List.BasicBlock {
        private final java.util.List/*<joeq.Compil3r.Quad.BasicBlock>*/ a;
        public BasicBlock(java.util.List/*<joeq.Compil3r.Quad.BasicBlock>*/ c) { this.a = c; }
        public int size() { return a.size(); }
        public Object get(int index) { return a.get(index); }
        public joeq.Compil3r.Quad.BasicBlock getBasicBlock(int index) { return (joeq.Compil3r.Quad.BasicBlock)a.get(index); }
        public void add(int i, Object o) { a.add(i, o); }
        public Object set(int i, Object o) { return a.set(i, o); }
        public Object remove(int i) { return a.remove(i); }
        public ListIterator.BasicBlock basicBlockIterator() { return new Iterator(a.listIterator()); }
        public static class Iterator implements ListIterator.BasicBlock {
            private java.util.ListIterator/*<joeq.Compil3r.Quad.BasicBlock>*/ i;
            public Iterator(java.util.ListIterator/*<joeq.Compil3r.Quad.BasicBlock>*/ l) { this.i = l; }
            public boolean hasNext() { return i.hasNext(); }
            public boolean hasPrevious() { return i.hasPrevious(); }
            public int nextIndex() { return i.nextIndex(); }
            public int previousIndex() { return i.previousIndex(); }
            public Object next() { return i.next(); }
            public joeq.Compil3r.Quad.BasicBlock nextBasicBlock() { return (joeq.Compil3r.Quad.BasicBlock)i.next(); }
            public Object previous() { return i.previous(); }
            public joeq.Compil3r.Quad.BasicBlock previousBasicBlock() { return (joeq.Compil3r.Quad.BasicBlock)i.previous(); }
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
            public joeq.Compil3r.Quad.BasicBlock nextBasicBlock() { throw new java.util.NoSuchElementException(); }
            public Object previous() { throw new java.util.NoSuchElementException(); }
            public joeq.Compil3r.Quad.BasicBlock previousBasicBlock() { throw new java.util.NoSuchElementException(); }
            public void remove() { throw new java.lang.IllegalStateException(); }
            public void set(Object o) { throw new java.lang.IllegalStateException(); }
            public void add(Object o) { throw new java.lang.UnsupportedOperationException(); }
            public static EmptyIterator INSTANCE = new EmptyIterator();
        }
    }
        
    public static class Quad extends java.util.AbstractList implements List.Quad {
        private final java.util.List/*<joeq.Compil3r.Quad.Quad>*/ a;
        public Quad(java.util.List/*<joeq.Compil3r.Quad.Quad>*/ c) { this.a = c; }
        public int size() { return a.size(); }
        public Object get(int index) { return a.get(index); }
        public joeq.Compil3r.Quad.Quad getQuad(int index) { return (joeq.Compil3r.Quad.Quad)a.get(index); }
        public void add(int i, Object o) { a.add(i, o); }
        public Object set(int i, Object o) { return a.set(i, o); }
        public Object remove(int i) { return a.remove(i); }
        public ListIterator.Quad quadIterator() { return new Iterator(a.listIterator()); }
        public static class Iterator implements ListIterator.Quad {
            private java.util.ListIterator/*<joeq.Compil3r.Quad.Quad>*/ i;
            public Iterator(java.util.ListIterator/*<joeq.Compil3r.Quad.Quad>*/ l) { this.i = l; }
            public boolean hasNext() { return i.hasNext(); }
            public boolean hasPrevious() { return i.hasPrevious(); }
            public int nextIndex() { return i.nextIndex(); }
            public int previousIndex() { return i.previousIndex(); }
            public Object next() { return i.next(); }
            public joeq.Compil3r.Quad.Quad nextQuad() { return (joeq.Compil3r.Quad.Quad)i.next(); }
            public Object previous() { return i.previous(); }
            public joeq.Compil3r.Quad.Quad previousQuad() { return (joeq.Compil3r.Quad.Quad)i.previous(); }
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
            public joeq.Compil3r.Quad.Quad nextQuad() { throw new java.util.NoSuchElementException(); }
            public Object previous() { throw new java.util.NoSuchElementException(); }
            public joeq.Compil3r.Quad.Quad previousQuad() { throw new java.util.NoSuchElementException(); }
            public void remove() { throw new java.lang.IllegalStateException(); }
            public void set(Object o) { throw new java.lang.IllegalStateException(); }
            public void add(Object o) { throw new java.lang.UnsupportedOperationException(); }
            public static EmptyIterator INSTANCE = new EmptyIterator();
        }
    }
        
    public static class ExceptionHandler extends java.util.AbstractList implements List.ExceptionHandler {
        private final java.util.List/*<joeq.Compil3r.Quad.ExceptionHandler>*/ a;
        public ExceptionHandler(java.util.List/*<joeq.Compil3r.Quad.ExceptionHandler>*/ c) { this.a = c; }
        public int size() { return a.size(); }
        public Object get(int index) { return a.get(index); }
        public joeq.Compil3r.Quad.ExceptionHandler getExceptionHandler(int index) { return (joeq.Compil3r.Quad.ExceptionHandler)a.get(index); }
        public void add(int i, Object o) { a.add(i, o); }
        public Object set(int i, Object o) { return a.set(i, o); }
        public Object remove(int i) { return a.remove(i); }
        public ListIterator.ExceptionHandler exceptionHandlerIterator() { return new Iterator(a.listIterator()); }
        public static class Iterator implements ListIterator.ExceptionHandler {
            private java.util.ListIterator/*<joeq.Compil3r.Quad.ExceptionHandler>*/ i;
            public Iterator(java.util.ListIterator/*<joeq.Compil3r.Quad.ExceptionHandler>*/ l) { this.i = l; }
            public boolean hasNext() { return i.hasNext(); }
            public boolean hasPrevious() { return i.hasPrevious(); }
            public int nextIndex() { return i.nextIndex(); }
            public int previousIndex() { return i.previousIndex(); }
            public Object next() { return i.next(); }
            public joeq.Compil3r.Quad.ExceptionHandler nextExceptionHandler() { return (joeq.Compil3r.Quad.ExceptionHandler)i.next(); }
            public Object previous() { return i.previous(); }
            public joeq.Compil3r.Quad.ExceptionHandler previousExceptionHandler() { return (joeq.Compil3r.Quad.ExceptionHandler)i.previous(); }
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
            public joeq.Compil3r.Quad.ExceptionHandler nextExceptionHandler() { throw new java.util.NoSuchElementException(); }
            public Object previous() { throw new java.util.NoSuchElementException(); }
            public joeq.Compil3r.Quad.ExceptionHandler previousExceptionHandler() { throw new java.util.NoSuchElementException(); }
            public void remove() { throw new java.lang.IllegalStateException(); }
            public void set(Object o) { throw new java.lang.IllegalStateException(); }
            public void add(Object o) { throw new java.lang.UnsupportedOperationException(); }
            public static EmptyIterator INSTANCE = new EmptyIterator();
        }
    }
}
