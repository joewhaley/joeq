/*
 * Created on Sep 20, 2003
 */
package Util.Collections;

import java.util.Iterator;

/**
 * @author jwhaley
 */
public interface IndexedMap {

    int get(Object o);
    Object get(int i);
    Iterator iterator();
    int size();
    
}
