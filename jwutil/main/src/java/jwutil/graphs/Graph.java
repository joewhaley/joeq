// Graph.java, created Jun 15, 2003 6:16:17 PM by joewhaley
// Copyright (C) 2003 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package jwutil.graphs;

import java.util.Collection;

/**
 * Graph
 * 
 * @author John Whaley
 * @version $Id$
 */
public interface Graph {

    Collection getRoots();

    Navigator getNavigator();

}
