/*
 * ExceptionHandlerIterator.java
 *
 * Created on April 22, 2001, 12:53 AM
 *
 */

package Compil3r.BytecodeAnalysis;

import java.util.List;
import java.util.ListIterator;
import java.util.NoSuchElementException;

import Util.AppendListIterator;

/*
 * @author  John Whaley
 * @version $Id$
 */
public class ExceptionHandlerIterator implements ListIterator {

    private final AppendListIterator iterator;
    
    /** Creates new ExceptionHandlerIterator */
    public ExceptionHandlerIterator(List exs, ExceptionHandlerSet parent) {
        ListIterator l2 = parent==null?null:parent.iterator();
        iterator = new AppendListIterator(exs.listIterator(), l2);
    }
    private ExceptionHandlerIterator() {
        iterator = null;
    }
    
    public boolean hasPrevious() { return iterator.hasPrevious(); }
    public boolean hasNext() { return iterator.hasNext(); }
    public Object previous() { return iterator.previous(); }
    public Object next() { return iterator.next(); }
    public int previousIndex() { return iterator.previousIndex(); }
    public int nextIndex() { return iterator.nextIndex(); }
    public void remove() { iterator.remove(); }
    public void set(Object o) { iterator.set(o); }
    public void add(Object o) { iterator.add(o); }
    
    public ExceptionHandler prevEH() { return (ExceptionHandler)previous(); }
    public ExceptionHandler nextEH() { return (ExceptionHandler)next(); }
    
    public static ExceptionHandlerIterator nullIterator() {
        return new ExceptionHandlerIterator() {
            public boolean hasPrevious() { return false; }
            public boolean hasNext() { return false; }
            public Object previous() { throw new NoSuchElementException(); }
            public Object next() { throw new NoSuchElementException(); }
            public int previousIndex() { return -1; }
            public int nextIndex() { return 0; }
            public void remove() { throw new IllegalStateException(); }
            public void set(Object o) { throw new IllegalStateException(); }
            public void add(Object o) { throw new UnsupportedOperationException(); }
            public ExceptionHandler prevEH() { throw new NoSuchElementException(); }
            public ExceptionHandler nextEH() { throw new NoSuchElementException(); }
        };
    }
}
