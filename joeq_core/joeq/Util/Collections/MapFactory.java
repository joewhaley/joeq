// MapFactory.java, created Tue Oct 19 22:42:28 1999 by pnkfelix
// Copyright (C) 1999 Felix S. Klock II <pnkfelix@mit.edu>
// Licensed under the terms of the GNU GPL; see COPYING for details.
package Util.Collections;

import java.util.Collections;
import java.util.Map;

/** <code>MapFactory</code> is a <code>Map</code> generator.
    Subclasses should implement constructions of specific types of
    <code>Map</code>s.
 
    @author  Felix S. Klock II <pnkfelix@mit.edu>
    @version $Id$
 */
public abstract class MapFactory {
    
    /** Creates a <code>MapFactory</code>. */
    public MapFactory() {
        
    }

    /** Generates a new, mutable, empty <code>Map</code>. */
    public final Map makeMap() {
        return this.makeMap(Collections.EMPTY_MAP);
    }

    /** Generates a new <code>Map</code>, using the entries of
        <code>map</code> as a template for its initial mappings. 
    */
    public abstract Map makeMap(Map map);

    
}
