package Util.Collections;

import java.util.Map;
import java.util.Set;

/**
 * A <code>MapSet</code> is a <code>java.util.Set</code> of
 * <code>Map.Entry</code>s which can also be accessed as a
 * <code>java.util.Map</code>.  Use the <code>entrySet()</code>
 * method of the <code>Map</code> to get back the <code>MapSet</code>.
 * 
 * @author  C. Scott Ananian <cananian@alumni.princeton.edu>
 * @version $Id$
 */
public interface MapSet/*<K,V>*/ extends Set/*<Map.Entry<K,V>>*/ {
    public Map/*<K,V>*/ asMap();
}
