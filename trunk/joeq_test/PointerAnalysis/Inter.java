// Inter.java, created Oct 14, 2003 5:50:20 PM by John Whaley
// Copyright (C) 2003 John Whaley
// Licensed under the terms of the GNU LGPL; see COPYING for details.

/**
 * Inter
 * 
 * @author John Whaley
 * @version $Id$
 */
public class Inter {

    final void finalmethod() {
        System.exit(1);
    }

    private final void privatefinalmethod() {
        System.exit(1);
    }

    private static void smethod(Inter j) {
        j.finalmethod();
        j.privatefinalmethod();
    }

    public static void main(String[] av) {
        smethod(new Inter());
    }

}
