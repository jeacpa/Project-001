import { Typography } from "@mui/material";
import { PageContainer } from "@toolpad/core";
import IssuesList from './IssuesList';


export default function SearchPage() {
  return (    
    <PageContainer breadcrumbs={[]}>
      <Typography>
        The issues here are taking from our github page and displayed here.  We need help writing the code to grab
        the issue from https://github.com/jeacpa/Project-001/issues.  I will presume we will use React to perform
        this.
      </Typography>
    </PageContainer>
  );
}
