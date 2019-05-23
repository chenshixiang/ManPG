function setduplicat_pduplicat(n)
 global  Dn pDn;
 Dn=sparse(DuplicationM(n));
 pDn=(Dn'*Dn)\Dn';
end