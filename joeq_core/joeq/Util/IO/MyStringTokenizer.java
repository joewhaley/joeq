// MyStringTokenizer.java, created Apr 21, 2004 7:06:28 PM 2004 by jwhaley
// Copyright (C) 2004 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Util.IO;

import java.util.Enumeration;
import java.util.NoSuchElementException;

/**
 * MyStringTokenizer is like StringTokenizer, but gives you access to
 * the string and position, and also ignores tokens inbetween quotation marks.
 */
public class MyStringTokenizer implements Enumeration
{
    public String getString() {
        return str;
    }
    
    public int getPosition() {
        return currentPosition;
    }
    
    private int currentPosition;
    private int newPosition;
    private int maxPosition;
    private String str;
    private String delimiters;
    private boolean retDelims;
    private boolean delimsChanged;

    private char maxDelimChar;

    private void setMaxDelimChar() {
        if (delimiters == null) {
            maxDelimChar = 0;
            return;
        }

        char m = 0;
        for (int i = 0; i < delimiters.length(); i++) {
            char c = delimiters.charAt(i);
            if (m < c)
            m = c;
        }
        maxDelimChar = m;
    }

    public MyStringTokenizer(String str, String delim, boolean returnDelims) {
        currentPosition = 0;
        newPosition = -1;
        delimsChanged = false;
        this.str = str;
        maxPosition = str.length();
        delimiters = delim;
        retDelims = returnDelims;
        setMaxDelimChar();
    }

    public MyStringTokenizer(String str, String delim) {
        this(str, delim, false);
    }

    public MyStringTokenizer(String str) {
        this(str, " \t\n\r\f");
    }

    private int skipDelimiters(int startPos) {
        if (delimiters == null)
            throw new NullPointerException();

        int position = startPos;
        while (!retDelims && position < maxPosition) {
            char c = str.charAt(position);
            if ((c > maxDelimChar) || (delimiters.indexOf(c) < 0))
                break;
            position++;
        }
        return position;
    }

    private int scanToken(int startPos) {
        int position = startPos;
        boolean inString = false;
        while (position < maxPosition) {
            char c = str.charAt(position);
            if (c == '"') {
                inString = !inString;
            } else
                if (!inString && ((c <= maxDelimChar) && (delimiters.indexOf(c) >= 0)))
                    break;
            position++;
        }
        if (retDelims && (startPos == position)) {
            char c = str.charAt(position);
            if ((c <= maxDelimChar) && (delimiters.indexOf(c) >= 0))
                position++;
        }
        return position;
    }

    public boolean hasMoreTokens() {
        newPosition = skipDelimiters(currentPosition);
        return (newPosition < maxPosition);
    }

    public String nextToken() {

        currentPosition = (newPosition >= 0 && !delimsChanged) ?  
            newPosition : skipDelimiters(currentPosition);

        delimsChanged = false;
        newPosition = -1;

        if (currentPosition >= maxPosition)
            throw new NoSuchElementException();
        int start = currentPosition;
        currentPosition = scanToken(currentPosition);
        return str.substring(start, currentPosition);
    }
    
    public String nextToken(String delim) {
        delimiters = delim;
        delimsChanged = true;
        setMaxDelimChar();
        return nextToken();
    }

    public boolean hasMoreElements() {
        return hasMoreTokens();
    }

    public Object nextElement() {
        return nextToken();
    }

    public int countTokens() {
        int count = 0;
        int currpos = currentPosition;
        while (currpos < maxPosition) {
                currpos = skipDelimiters(currpos);
            if (currpos >= maxPosition)
            break;
                currpos = scanToken(currpos);
            count++;
        }
        return count;
    }
    
}
