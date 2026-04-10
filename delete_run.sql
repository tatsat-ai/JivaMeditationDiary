DELETE FROM Similarities 
WHERE section_a IN (SELECT id FROM Sections WHERE run_id = 4)
   OR section_b IN (SELECT id FROM Sections WHERE run_id = 4);

DELETE FROM Sections WHERE run_id = 4;

DELETE FROM Sessions WHERE run_id = 4;