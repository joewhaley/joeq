// StackTraceElement.java, created Tue Aug  6 14:33:23 2002 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package ClassLib.sun14_win32.java.lang;

/*
 * @author  John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public final class StackTraceElement {
    private java.lang.String declaringClass;
    private java.lang.String methodName;
    private java.lang.String fileName;
    private int lineNumber;

    StackTraceElement(java.lang.String declaringClass,
                      java.lang.String methodName,
                      java.lang.String fileName,
                      int lineNumber) {
        this.declaringClass = declaringClass;
        this.methodName = methodName;
        this.fileName = fileName;
        this.lineNumber = lineNumber;
    }
}
