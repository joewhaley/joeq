// PredicateWrapper.java, created Thu Feb 24 15:56:13 2000 by salcianu
// Copyright (C) 2000 Alexandru SALCIANU <salcianu@MIT.EDU>
// Licensed under the terms of the GNU GPL; see COPYING for details.
package Util;

/**
 * <code>PredicateWrapper</code> wraps a predicate on an <code>Object</code>.
 This is supposed to allow us to send predicate functions as arguments to
 high-level functions.
 * 
 * @author  Alexandru SALCIANU <salcianu@MIT.EDU>
 * @version $Id$
 */
public interface PredicateWrapper {
    public boolean check(Object obj);
}
