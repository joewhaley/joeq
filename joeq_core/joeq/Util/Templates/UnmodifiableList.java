package Util.Templates;

import Util.Collections.UnmodifiableListIterator;

/**
 * @author John Whaley
 * @version $Id$
 */
public abstract class UnmodifiableList {
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
