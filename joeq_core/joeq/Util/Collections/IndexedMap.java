/*
 * Created on Sep 20, 2003
 */
package joeq.Util.Collections;

import java.util.Iterator;

/**
 * @author jwhaley
 */
public interface IndexedMap {

    int get(Object o);
    Object get(int i);
    boolean contains(Object o);
    Iterator iterator();
    int size();
    
}
