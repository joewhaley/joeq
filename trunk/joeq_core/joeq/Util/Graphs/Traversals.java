// Traversals.java, created Thu May 26 23:09:37 2003 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package Util.Graphs;

import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

/**
 * @author  John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public abstract class Traversals {
    
    public static List preOrder(Navigator nav, Object root) {
        return traversal_helper(nav, Collections.singleton(root), PREORDER);
    }
    public static List preOrder(Navigator nav, Collection roots) {
        return traversal_helper(nav, roots, PREORDER);
    }
    
    public static List reversePreOrder(Navigator nav, Object root) {
        return traversal_helper(nav, Collections.singleton(root), REVERSE_PREORDER);
    }
    public static List reversePreOrder(Navigator nav, Collection roots) {
        return traversal_helper(nav, roots, REVERSE_PREORDER);
    }
    
    public static List inOrder(Navigator nav, Object root) {
        return traversal_helper(nav, Collections.singleton(root), INORDER);
    }
    public static List inOrder(Navigator nav, Collection roots) {
        return traversal_helper(nav, roots, INORDER);
    }
    
    public static List reverseInOrder(Navigator nav, Object root) {
        return traversal_helper(nav, Collections.singleton(root), REVERSE_INORDER);
    }
    public static List reverseInOrder(Navigator nav, Collection roots) {
        return traversal_helper(nav, roots, REVERSE_INORDER);
    }
    
    public static List postOrder(Navigator nav, Object root) {
        return traversal_helper(nav, Collections.singleton(root), POSTORDER);
    }
    public static List postOrder(Navigator nav, Collection roots) {
        return traversal_helper(nav, roots, POSTORDER);
    }
    
    public static List reversePostOrder(Navigator nav, Object root) {
        return traversal_helper(nav, Collections.singleton(root), REVERSE_POSTORDER);
    }
    public static List reversePostOrder(Navigator nav, Collection roots) {
        return traversal_helper(nav, roots, REVERSE_POSTORDER);
    }
    
    private static final byte PREORDER          = 1;
    private static final byte REVERSE_PREORDER  = 2;
    private static final byte INORDER           = 3;
    private static final byte REVERSE_INORDER   = 4;
    private static final byte POSTORDER         = 5;
    private static final byte REVERSE_POSTORDER = 6;
    
    private static final List traversal_helper(Navigator nav, Collection roots,
                                               byte type) {
        HashSet visited = new HashSet();
        LinkedList result = new LinkedList();
        for (Iterator i=roots.iterator(); i.hasNext(); ) {
            Object root = i.next();
            traversal_helper(nav, root, visited, result, type);
        }
        return result;
    }
    
    /** Helper function to compute reverse post order. */
    private static final void traversal_helper(
        Navigator nav,
        Object node,
        HashSet visited,
        LinkedList result,
        byte type) {
        if (visited.contains(node)) return; visited.add(node);
        if (type == PREORDER) result.add(node);
        else if (type == REVERSE_PREORDER) result.addFirst(node);
        Collection bbs = nav.next(node);
        Iterator bbi = bbs.iterator();
        while (bbi.hasNext()) {
            Object node2 = bbi.next();
            traversal_helper(nav, node2, visited, result, type);
            if (type == INORDER) {
                result.add(node);
                type = 0;
            } else if (type == REVERSE_INORDER) {
                result.addFirst(node);
                type = 0;
            }
        }
        if (type == POSTORDER) result.add(node);
        else if (type == REVERSE_POSTORDER) result.addFirst(node);
    }
    
}
